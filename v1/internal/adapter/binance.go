package adapter

import (
	"encoding/json"
	"fmt"
	"singularity/internal/core"
	"strconv"
	"strings"
	"time"
)

type BinanceAdapter struct {
	*BaseWSClient
	stream chan core.NormalizedMessage
}

func NewBinanceAdapter() *BinanceAdapter {
	// Use Futures WebSocket for Liquidations
	url := "wss://fstream.binance.com/ws"
	client := NewBaseWSClient("Binance", url)
	
	return &BinanceAdapter{
		BaseWSClient: client,
		stream:       make(chan core.NormalizedMessage, 65536), // Increased buffer to 64k for HFT bursts
	}
}

func (b *BinanceAdapter) Subscribe(symbols []string) error {
	params := make([]string, 0)
	for _, s := range symbols {
		s = strings.ToLower(s)
		params = append(params, s+"@aggTrade")
		params = append(params, s+"@depth20@100ms")
		params = append(params, s+"@forceOrder")
	}

	payload := map[string]interface{}{
		"method": "SUBSCRIBE",
		"params": params,
		"id":     time.Now().Unix(),
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	b.SendChan <- data
	return nil
}

func (b *BinanceAdapter) Stream() <-chan core.NormalizedMessage {
	// Start a goroutine to parse messages from BaseWSClient.ReadChan
	go func() {
		count := 0
		for raw := range b.ReadChan {
			count++
			if count % 100 == 0 {
				fmt.Printf("DEBUG: Received %d msgs. Last raw len: %d\n", count, len(raw))
			}
			msg, err := b.parse(raw)
			if err != nil {
				fmt.Printf("DEBUG: Parse error: %v\n", err)
			}
			if err == nil && msg != nil {
				select {
				case b.stream <- *msg:
				default:
					// Drop message if channel is full to prevent deadlocking the pump
				}
			} else if msg == nil && err == nil {
				// Ignored event
				// fmt.Println("Ignored event")
			}
		}
		close(b.stream)
	}()
	return b.stream
}

// Internal structures for Binance JSON
type binanceAggTrade struct {
	E interface{} `json:"E"` // Event time (mixed types?)
	S string      `json:"s"` // Symbol
	P string      `json:"p"` // Price
	Q string      `json:"q"` // Quantity
	M bool        `json:"m"` // Is buyer maker?
}

type binanceDepth struct {
	E interface{}     `json:"E"`
	S string          `json:"s"`
	B [][]interface{} `json:"bids"`
	A [][]interface{} `json:"asks"`
}

func (b *BinanceAdapter) parse(raw []byte) (*core.NormalizedMessage, error) {
	// Parse event type
	var quick struct {
		Event     string      `json:"e"`
		EventTime interface{} `json:"E"` // Capture E to prevent it matching 'e'
	}
	
	if err := json.Unmarshal(raw, &quick); err != nil {
		// Log raw message if it fails, so we can debug
		// Limit length to avoid console spam
		s := string(raw)
		if len(s) > 100 { s = s[:100] + "..." }
		return nil, fmt.Errorf("%v | Raw: %s", err, s)
	}

	if quick.Event == "aggTrade" {
		var t binanceAggTrade
		if err := json.Unmarshal(raw, &t); err != nil {
			return nil, err
		}
		
		price, _ := strconv.ParseFloat(t.P, 64)
		qty, _ := strconv.ParseFloat(t.Q, 64)
		
		side := core.Buy
		if t.M {
			side = core.Sell
		}

		trade := core.Trade{
			Symbol:    t.S,
			Price:     price,
			Size:      qty,
			Side:      side,
			Timestamp: toInt64(t.E) * 1000, 
		}

		return &core.NormalizedMessage{
			Type:       "trade",
			Payload:    trade,
			ReceivedAt: time.Now(),
		}, nil

	} else if quick.Event == "forceOrder" {
		// Parse Liquidation
		var liq struct {
			Order struct {
				S string `json:"s"` // Symbol
				S_side string `json:"S"` // Side
				P string `json:"p"` // Price
				Q string `json:"q"` // Quantity
				T int64  `json:"T"` // Order Trade Time
			} `json:"o"`
		}
		if err := json.Unmarshal(raw, &liq); err != nil {
			return nil, err
		}

		price, _ := strconv.ParseFloat(liq.Order.P, 64)
		qty, _ := strconv.ParseFloat(liq.Order.Q, 64)
		side := core.Buy
		if liq.Order.S_side == "SELL" {
			side = core.Sell
		}

		return &core.NormalizedMessage{
			Type: "liq",
			Payload: core.Trade{
				Symbol:    liq.Order.S,
				Price:     price,
				Size:      qty,
				Side:      side,
				Timestamp: liq.Order.T * 1000,
				IsLiquidation: true,
			},
			ReceivedAt: time.Now(),
		}, nil

	} else if quick.Event == "depthUpdate" || quick.Event == "depth20" {
		var d binanceDepth
		if err := json.Unmarshal(raw, &d); err != nil {
			return nil, err
		}

		// If d.S (Symbol) is missing (e.g. depth20 might not have it inside?), check payload
		// depth20 payload: { "lastUpdateId": ..., "bids": ..., "asks": ... }
		// It has NO Event Type "e". So quick.Event is empty.
		// We shouldn't be here if quick.Event is matched.
		// Wait, if it has no 'e', quick.Event is "".
		// So this block won't execute.
		
		// If we want to support depth20 snapshot (which has no 'e'), we need to detect it differently.
		// e.g. check if "bids" exists?
		
		book := core.OrderBookL2{
			Symbol:    d.S,
			Timestamp: toInt64(d.E) * 1000,
			Bids:      parseDepthLevels(d.B),
			Asks:      parseDepthLevels(d.A),
		}

		return &core.NormalizedMessage{
			Type:       "book",
			Payload:    book,
			ReceivedAt: time.Now(),
		}, nil
	} 
	
	// Check for Snapshot (No Event Type)
	if quick.Event == "" {
		// Try to parse as depth snapshot
		// It usually has "lastUpdateId" and "bids"
		var snap struct {
			LastUpdateId int64           `json:"lastUpdateId"`
			Bids         [][]interface{} `json:"bids"`
			Asks         [][]interface{} `json:"asks"`
		}
		if err := json.Unmarshal(raw, &snap); err == nil && len(snap.Bids) > 0 {
			// Snapshot!
			// But we don't know the symbol and timestamp from the payload usually... 
			// Wait, stream wrapper has stream name? 
			// For now, we ignore snapshot or infer?
			// The user wants Data Collection. AggTrade is most important for Features (Price/Velocity).
			// Book is for Resistance.
			// Let's ignore Snapshots for now if they are hard to parse without context.
			return nil, nil // Ignore
		}
	}

	return nil, nil // Ignored
}

func toInt64(v interface{}) int64 {
	switch val := v.(type) {
	case float64:
		return int64(val)
	case string:
		i, _ := strconv.ParseInt(val, 10, 64)
		return i
	case int64:
		return val
	default:
		return 0
	}
}

func parseDepthLevels(list [][]interface{}) []core.OrderBookLevel {
	res := make([]core.OrderBookLevel, len(list))
	for i, item := range list {
		// item[0] is price string, item[1] is result string
		pStr := item[0].(string)
		qStr := item[1].(string)
		p, _ := strconv.ParseFloat(pStr, 64)
		q, _ := strconv.ParseFloat(qStr, 64)
		res[i] = core.OrderBookLevel{Price: p, Size: q}
	}
	return res
}
