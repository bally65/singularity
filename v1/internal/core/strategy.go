package core

// StrategyConfig defines thresholds for different trading regimes
type StrategyConfig struct {
	SignalThreshold float64 // Minimum AI confidence to trigger signal
	RiskVaRPercent  float64 // Value-at-Risk limit
	MaxPositionSize float64 // Maximum size per trade
}

// OrderStatus defines the lifecycle of a trade
type OrderStatus string

const (
	OrderPending   OrderStatus = "PENDING"
	OrderFilled    OrderStatus = "FILLED"
	OrderCancelled OrderStatus = "CANCELLED"
	OrderRejected  OrderStatus = "REJECTED"
)

// ExecutedOrder represents a real or paper trade result
type ExecutedOrder struct {
	OrderID    string
	Symbol     string
	Side       Direction
	Price      float64
	Qty        float64
	Status     OrderStatus
	Timestamp  int64
	Commission float64
}

// PortfolioState tracks current holdings and balance
type PortfolioState struct {
	Balance    float64
	Positions  map[string]float64
	Equity     float64
	LastUpdate int64
}
