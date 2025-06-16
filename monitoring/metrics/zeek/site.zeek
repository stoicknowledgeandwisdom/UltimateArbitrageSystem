# Zeek configuration for arbitrage system network monitoring

# Load core Zeek modules
@load base/protocols/conn
@load base/protocols/dns
@load base/protocols/http
@load base/protocols/ssl
@load base/protocols/ssh
@load base/protocols/smtp
@load base/frameworks/notice
@load base/frameworks/intel
@load base/frameworks/logging
@load base/frameworks/software
@load base/frameworks/files
@load base/utils/site

# Security-focused modules
@load policy/protocols/ssl/weak-keys
@load policy/protocols/ssl/known-certs
@load policy/protocols/http/detect-sqli
@load policy/protocols/http/detect-webapps
@load policy/frameworks/intel/seen
@load policy/frameworks/intel/do_notice
@load policy/integration/collective-intel
@load policy/frameworks/notice/extend-email

# Custom security monitoring
@load ./custom-detections

# Network interfaces to monitor
redef interface = "eth0";

# Define internal networks for the arbitrage system
redef Site::local_nets = {
    10.0.0.0/8,
    192.168.0.0/16,
    172.16.0.0/12
};

# Email configuration for alerts
redef Notice::mail_dest = "security@arbitrage-system.com";
redef Notice::sendmail = "/usr/sbin/sendmail";

# Logging configuration
redef Log::default_rotation_interval = 1hr;
redef Log::default_rotation_postprocessor_cmd = "gzip";

# Intel framework configuration
redef Intel::read_files += {
    "/opt/zeek/share/zeek/intel/indicators.dat"
};

# Custom notice types for trading system
export {
    redef enum Notice::Type += {
        TradingSystem::SuspiciousAPI,
        TradingSystem::UnauthorizedAccess,
        TradingSystem::DataExfiltration,
        TradingSystem::AnomalousTraffic,
        TradingSystem::ExchangeConnectivity,
        TradingSystem::SecurityViolation
    };
}

# HTTP monitoring for API endpoints
event http_request(c: connection, method: string, original_URI: string, unescaped_URI: string, version: string) {
    # Monitor API calls to trading endpoints
    if (/api\/v[0-9]+\/(trade|order|portfolio|risk)/ in unescaped_URI) {
        # Log high-frequency trading API calls
        if (method in set("POST", "PUT", "DELETE")) {
            NOTICE([$note=TradingSystem::SuspiciousAPI,
                   $msg=fmt("Trading API call: %s %s from %s", method, unescaped_URI, c$id$orig_h),
                   $conn=c]);
        }
    }
    
    # Detect potential SQL injection
    if (/union|select|insert|update|delete|drop|create|alter/i in unescaped_URI) {
        NOTICE([$note=TradingSystem::SecurityViolation,
               $msg=fmt("Potential SQL injection: %s", unescaped_URI),
               $conn=c]);
    }
}

# SSL/TLS monitoring
event ssl_established(c: connection) {
    # Monitor certificate details for exchange connections
    if (c$ssl?$subject && /exchange|trading|crypto/ in c$ssl$subject) {
        print fmt("SSL connection to exchange: %s -> %s (Subject: %s)", 
                 c$id$orig_h, c$id$resp_h, c$ssl$subject);
    }
}

# DNS monitoring for potential data exfiltration
event dns_request(c: connection, msg: dns_msg, query: string, qtype: count, qclass: count) {
    # Monitor for suspicious DNS queries
    if (|query| > 50 || /[0-9a-f]{32,}/ in query) {
        NOTICE([$note=TradingSystem::DataExfiltration,
               $msg=fmt("Suspicious DNS query: %s", query),
               $conn=c]);
    }
}

# Connection monitoring
event connection_established(c: connection) {
    # Monitor connections to known exchange endpoints
    local exchanges = set(
        "api.binance.com",
        "api.coinbase.com", 
        "api.kraken.com",
        "ftx.com",
        "api.bitfinex.com"
    );
    
    if (addr_to_hostname(c$id$resp_h) in exchanges) {
        print fmt("Exchange connection established: %s -> %s", 
                 c$id$orig_h, addr_to_hostname(c$id$resp_h));
    }
}

# File transfer monitoring
event file_new(f: fa_file) {
    # Monitor large file transfers that might indicate data theft
    if (f?$total_bytes && f$total_bytes > 100000000) { # 100MB
        NOTICE([$note=TradingSystem::DataExfiltration,
               $msg=fmt("Large file transfer detected: %d bytes", f$total_bytes),
               $file=f]);
    }
}

# Custom script for trading system-specific monitoring
function monitor_trading_traffic(c: connection): bool {
    # Check for patterns specific to trading systems
    local is_trading = F;
    
    # Monitor specific ports used by trading systems
    if (c$id$resp_p in set(8000/tcp, 8001/tcp, 8002/tcp, 9999/tcp)) {
        is_trading = T;
    }
    
    # Monitor WebSocket connections (common for real-time trading data)
    if (c$service?$ssl && "websocket" in c$service) {
        is_trading = T;
    }
    
    return is_trading;
}

# Rate limiting detection
global connection_counts: table[addr] of count &default=0 &create_expire=1min;

event new_connection(c: connection) {
    ++connection_counts[c$id$orig_h];
    
    # Alert on high connection rates (potential DDoS or scanning)
    if (connection_counts[c$id$orig_h] > 100) {
        NOTICE([$note=TradingSystem::AnomalousTraffic,
               $msg=fmt("High connection rate from %s: %d connections/minute", 
                       c$id$orig_h, connection_counts[c$id$orig_h]),
               $conn=c]);
    }
}

# Geographic anomaly detection
event connection_state_remove(c: connection) {
    # Monitor connections from unusual geographic locations
    # This would integrate with GeoIP databases
    if (c$orig?$country_code && c$orig$country_code !in set("US", "GB", "DE", "JP", "SG")) {
        if (monitor_trading_traffic(c)) {
            NOTICE([$note=TradingSystem::AnomalousTraffic,
                   $msg=fmt("Trading connection from unusual location: %s (%s)", 
                           c$id$orig_h, c$orig$country_code),
                   $conn=c]);
        }
    }
}

