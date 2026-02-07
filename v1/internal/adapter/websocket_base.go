package adapter

import (
	"context"
	"errors"
	"log"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// BaseWSClient handles the lifecycle of a single WebSocket connection.
// It manages the Read/Write pumps and reconnection logic.
type BaseWSClient struct {
	Name string
	URL  string

	// Connection state
	conn   *websocket.Conn
	mu     sync.Mutex
	done   chan struct{} // Signal to stop pumps
	ctx    context.Context
	cancel context.CancelFunc

	// Configuration
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	PingInterval time.Duration

	// Channels
	SendChan chan []byte // Outgoing raw messages (subscriptions/pings)
	ReadChan chan []byte // Incoming raw messages
	ErrChan  chan error  // async errors
}

func NewBaseWSClient(name, url string) *BaseWSClient {
	return &BaseWSClient{
		Name:         name,
		URL:          url,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 10 * time.Second,
		PingInterval: 20 * time.Second,
		SendChan:     make(chan []byte, 256), // Buffered to prevent blocking
		ReadChan:     make(chan []byte, 1024), // Larger buffer for high-throughput ingress
		ErrChan:      make(chan error, 10),
	}
}

// Connect starts the connection loop. It is blocking if we strictly want to wait for first connect,
// but here we launch the manager loop.
func (c *BaseWSClient) Connect(ctx context.Context) error {
	c.ctx, c.cancel = context.WithCancel(ctx)
	c.done = make(chan struct{})

	// Initial Dial
	if err := c.dial(); err != nil {
		return err
	}

	// Start Pumps
	go c.readPump()
	go c.writePump()

	// Reconnection Monitor can be added here or externally
	return nil
}

func (c *BaseWSClient) dial() error {
	log.Printf("[%s] Connecting to %s...", c.Name, c.URL)
	conn, _, err := websocket.DefaultDialer.Dial(c.URL, nil)
	if err != nil {
		return err
	}
	c.mu.Lock()
	c.conn = conn
	c.mu.Unlock()
	log.Printf("[%s] Connected.", c.Name)
	return nil
}

func (c *BaseWSClient) Close() {
	if c.cancel != nil {
		c.cancel()
	}
	close(c.done)
	if c.conn != nil {
		c.conn.Close()
	}
}

// writePump dumps messages from SendChan to the websocket
func (c *BaseWSClient) writePump() {
	ticker := time.NewTicker(c.PingInterval)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case <-c.done:
			return
		case <-c.ctx.Done():
			return
		case msg := <-c.SendChan:
			c.conn.SetWriteDeadline(time.Now().Add(c.WriteTimeout))
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				c.ErrChan <- err
				return // Reconnect triggered by readPump failure usually, or handle here
			}
		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(c.WriteTimeout))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// readPump pumps messages from websocket to ReadChan
func (c *BaseWSClient) readPump() {
	lastMsg := time.Now()
	ticker := time.NewTicker(5 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	c.conn.SetReadLimit(1024 * 1024 * 5) // 5MB limit
	c.conn.SetReadDeadline(time.Now().Add(c.ReadTimeout))
	
	c.conn.SetPongHandler(func(string) error { 
		c.conn.SetReadDeadline(time.Now().Add(c.ReadTimeout))
		return nil 
	})

	// Monitor Watchdog
	go func() {
		for range ticker.C {
			if time.Since(lastMsg) > 15*time.Second {
				log.Printf("[%s] WATCHDOG: No data for 15s, triggering reconnect...", c.Name)
				c.conn.Close() // Force readPump to exit and reconnect
				return
			}
		}
	}()

	for {
		select {
		case <-c.done:
			return
		case <-c.ctx.Done():
			return
		default:
			_, message, err := c.conn.ReadMessage()
			if err != nil {
				c.ErrChan <- errors.New("connection closed")
				return
			}
			lastMsg = time.Now()
			c.ReadChan <- message
		}
	}
}
