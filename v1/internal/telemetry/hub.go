package telemetry

import (
	"log"
	"net/http"
	"sync"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool { return true },
}

type Hub struct {
	clients map[*websocket.Conn]bool
	broadcast chan []byte
	lock sync.Mutex
}

func NewHub() *Hub {
	return &Hub{
		clients:   make(map[*websocket.Conn]bool),
		broadcast: make(chan []byte),
	}
}

func (h *Hub) Run() {
	for {
		message := <-h.broadcast
		h.lock.Lock()
		for client := range h.clients {
			err := client.WriteMessage(websocket.TextMessage, message)
			if err != nil {
				client.Close()
				delete(h.clients, client)
			}
		}
		h.lock.Unlock()
	}
}

func (h *Hub) Broadcast(msg []byte) {
	h.broadcast <- msg
}

func StartServer(hub *Hub, addr string) {
	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("WS Upgrade Error:", err)
			return
		}
		hub.lock.Lock()
		hub.clients[conn] = true
		hub.lock.Unlock()
	})

	log.Printf("Telemetry Server listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal("Telemetry Server Error:", err)
	}
}
