package telemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// DiscordNotifier sends alerts to a Discord Webhook
type DiscordNotifier struct {
	webhookURL string
	enabled    bool
}

func NewDiscordNotifier(webhookURL string) *DiscordNotifier {
	return &DiscordNotifier{
		webhookURL: webhookURL,
		enabled:    webhookURL != "",
	}
}

func (d *DiscordNotifier) SendAlert(title, message string, color int) error {
	if !d.enabled {
		return nil
	}

	payload := map[string]interface{}{
		"embeds": []map[string]interface{}{
			{
				"title":       title,
				"description": message,
				"color":       color,
				"footer": map[string]string{
					"text": "Singularity Engine | High-Frequency AI Alert",
				},
				"timestamp": time.Now().Format(time.RFC3339),
			},
		},
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	resp, err := http.Post(d.webhookURL, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("discord returned status: %d", resp.StatusCode)
	}

	return nil
}
