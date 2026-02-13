use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use url::Url;
use chrono::Utc;
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
struct BinanceTrade {
    #[serde(rename = "s")]
    symbol: String,
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
    println!("🚀 Starting Singularity Engine V2 Multi-Asset (Rust)...");

    // Subscribing to BTC, ETH, and SOL
    let url = Url::parse("wss://fstream.binance.com/ws/btcusdt@trade/ethusdt@trade/solusdt@trade").unwrap();
    
    loop {
        println!("📡 Connecting to Binance Multi-Stream...");
        match connect_async(url.clone()).await {
            Ok((mut ws_stream, _)) => {
                println!("✅ Connected.");
                
                // We'll keep symbols in separate files for easier processing by Python scripts
                let mut files = HashMap::new();

                while let Some(msg) = ws_stream.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(trade) = serde_json::from_str::<BinanceTrade>(&text) {
                                let now = Utc::now().timestamp_millis();
                                let filename = format!("dataset_{}.csv", trade.symbol.to_lowercase());
                                
                                let file = files.entry(trade.symbol.clone()).or_insert_with(|| {
                                    OpenOptions::new()
                                        .create(true)
                                        .append(true)
                                        .open(&filename)
                                        .unwrap()
                                });

                                let line = format!("{},{},{},{},{}\n", 
                                    now, trade.event_time, trade.price, trade.quantity, trade.is_buyer_maker);
                                if let Err(e) = file.write_all(line.as_bytes()) {
                                    eprintln!("❌ Write error for {}: {}", trade.symbol, e);
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
