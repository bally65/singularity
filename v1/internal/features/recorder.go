package features

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"time"
)

// DataPoint represents a snapshot in time
type DataPoint struct {
	Timestamp int64
	Price     float64
	Vector    *FeatureVector
}

type Recorder struct {
	file       *os.File
	writer     *csv.Writer
	buffer     []DataPoint
	targetTime time.Duration // e.g., 60 seconds
	
	// New Async Storage Props
	writeChan  chan []string
	done       chan struct{}
}

func NewRecorder(filename string, lookAhead time.Duration) *Recorder {
	// ... (Existing Stat check) ...
	fileInfo, err := os.Stat(filename)
	needHeader := os.IsNotExist(err) || fileInfo.Size() == 0

	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open dataset file: %v", err)
	}

	w := csv.NewWriter(f)
	
	if needHeader {
		header := []string{
			"timestamp", "price", 
			"velocity", "accel", "entropy", "mass", "imbalance", "liq_force", "whale_force",
			"label_return_60s", "label_class_60s", 
		}
		w.Write(header)
		w.Flush()
	}

	r := &Recorder{
		file:       f,
		writer:     w,
		buffer:     make([]DataPoint, 0, 10000),
		targetTime: lookAhead,
		writeChan:  make(chan []string, 50000), // Massive buffer for async writing
		done:       make(chan struct{}),
	}

	// Start Background Writer Loop
	go r.backgroundWriter()

	return r
}

// AddSnapshot records the current state
func (r *Recorder) AddSnapshot(v *FeatureVector, price float64) {
	r.buffer = append(r.buffer, DataPoint{
		Timestamp: v.Timestamp, // Unix Micro
		Price:     price,
		Vector:    v,
	})
}

// Process checks for mature data points that can now be labeled
func (r *Recorder) Process(currentPrice float64, currentTime int64) {
	validIndex := -1
	targetMicro := r.targetTime.Microseconds()

	for i, point := range r.buffer {
		age := currentTime - point.Timestamp
		if age >= targetMicro {
			r.writeRow(point, currentPrice)
			validIndex = i
		} else {
			break
		}
	}

	if validIndex >= 0 {
		r.buffer = r.buffer[validIndex+1:]
	}
}

func (r *Recorder) backgroundWriter() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case row := <-r.writeChan:
			r.writer.Write(row)
		case <-ticker.C:
			r.writer.Flush() // Periodic flush to disk
		case <-r.done:
			r.writer.Flush()
			return
		}
	}
}

// ... AddSnapshot stays same ...

func (r *Recorder) writeRow(p DataPoint, futurePrice float64) {
	priceChange := (futurePrice - p.Price) / p.Price
	
	labelClass := "0"
	threshold := 0.0005 
	if priceChange > threshold {
		labelClass = "1"
	} else if priceChange < -threshold {
		labelClass = "-1"
	}

	row := []string{
		fmt.Sprintf("%d", p.Timestamp),
		fmt.Sprintf("%.2f", p.Price),
		fmt.Sprintf("%.4f", p.Vector.PriceVelocity),
		fmt.Sprintf("%.4f", p.Vector.PriceAccel),
		fmt.Sprintf("%.4f", p.Vector.Entropy),
		fmt.Sprintf("%.4f", p.Vector.MassResistance),
		fmt.Sprintf("%.4f", p.Vector.OrderImbalance),
		fmt.Sprintf("%.4f", p.Vector.NetForce),
		fmt.Sprintf("%.4f", p.Vector.WhaleForce),
		fmt.Sprintf("%.6f", priceChange),
		labelClass,
	}
	
	// Send to async channel instead of writing directly
	select {
	case r.writeChan <- row:
	default:
		// If channel is totally full, we might have an IO bottleneck
		// log.Println("Warning: Recorder write channel full, dropping data row")
	}
}

func (r *Recorder) Close() {
	close(r.done)
	r.file.Close()
}
