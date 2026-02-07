package core

import "time"

// Direction represents the side of a trade or order (Buy/Sell)
type Direction int8

const (
	Buy  Direction = 1
	Sell Direction = -1
)

// MarketData represents the unified binary structure we aim for.
// To optimize for cache locality and CGO zero-copy, we use fixed-size types where possible.
// For the initial Go implementation, we use native types for ease of math.

// Trade represents a single executed trade
type Trade struct {
	Symbol    string    `json:"s"`
	Price     float64   `json:"p"`
	Size      float64   `json:"q"`
	Side      Direction `json:"S"`
	Timestamp int64     `json:"t"` // Unix Microseconds
	IsLiquidation bool  `json:"l,omitempty"` // Derived field
}

// OrderBookLevel represents a single price depth
type OrderBookLevel struct {
	Price float64
	Size  float64
}

// OrderBookL2 represents a snapshot or delta of the book
type OrderBookL2 struct {
	Symbol    string
	Timestamp int64
	Bids      []OrderBookLevel // Sorted High to Low
	Asks      []OrderBookLevel // Sorted Low to High
}

// NormalizedMessage is the container passed through channels
type NormalizedMessage struct {
	Type      string // "trade", "book", "liq"
	Payload   interface{}
	ReceivedAt time.Time
}
