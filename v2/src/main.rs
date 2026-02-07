use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use url::Url;
use chrono::Utc;

#[derive(Debug, Serialize, Deserialize)]
struct BinanceTrade {
    #[serde(rename = "E")]
    event_time: u64,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "m")]
    is_buyer_maker: bool,
}

#[tokio::main]
async fn main() {
    env_logger::init();
    println!("🚀 Starting Singularity Engine V2 (Rust)...");

    let url = Url::parse("wss://fstream.binance.com/ws/btcusdt@trade").unwrap();
    
    loop {
        println!("📡 Connecting to Binance...");
        match connect_async(url.clone()).await {
            Ok((mut ws_stream, _)) => {
                println!("✅ Connected.");
                
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("dataset_v2.csv")
                    .unwrap();

                while let Some(msg) = ws_stream.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(trade) = serde_json::from_str::<BinanceTrade>(&text) {
                                let now = Utc::now().timestamp_millis();
                                let line = format!("{},{},{},{},{}\n", 
                                    now, trade.event_time, trade.price, trade.quantity, trade.is_buyer_maker);
                                if let Err(e) = file.write_all(line.as_bytes()) {
                                    eprintln!("❌ Write error: {}", e);
                                }
                            }
                        }
                        Ok(Message::Close(_)) => {
                            println!("⚠️ Connection closed by server.");
                            break;
                        }
                        Err(e) => {
                            eprintln!("❌ WebSocket error: {}", e);
                            break;
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Connection failed: {}. Retrying in 5s...", e);
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        }
    }
}
