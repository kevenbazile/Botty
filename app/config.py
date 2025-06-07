import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str = 'kraken'
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = False
    fee_rate: float = 0.001

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_balance: float = 10000.0
    fee_rate: float = 0.001
    start_date: str = '2024-01-01'
    end_date: str = '2024-12-31'
    timeframe: str = '1h'
    symbol: str = 'BTC/USD'

@dataclass
class StrategyConfig:
    """Strategy-specific configuration"""
    name: str = 'simple_strategy'
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class RiskManagementConfig:
    """Risk management settings"""
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_percent: float = 0.05  # 5% stop loss
    take_profit_percent: float = 0.10  # 10% take profit
    max_drawdown_percent: float = 0.20  # 20% max drawdown

@dataclass
class TradingConfig:
    """Main trading configuration"""
    exchange: ExchangeConfig = None
    backtest: BacktestConfig = None
    strategy: StrategyConfig = None
    risk_management: RiskManagementConfig = None
    
    def __post_init__(self):
        if self.exchange is None:
            self.exchange = ExchangeConfig()
        if self.backtest is None:
            self.backtest = BacktestConfig()
        if self.strategy is None:
            self.strategy = StrategyConfig()
        if self.risk_management is None:
            self.risk_management = RiskManagementConfig()

class ConfigManager:
    """Configuration manager for loading/saving configs"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = TradingConfig()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Load each section
                if 'exchange' in data:
                    self.config.exchange = ExchangeConfig(**data['exchange'])
                if 'backtest' in data:
                    self.config.backtest = BacktestConfig(**data['backtest'])
                if 'strategy' in data:
                    self.config.strategy = StrategyConfig(**data['strategy'])
                if 'risk_management' in data:
                    self.config.risk_management = RiskManagementConfig(**data['risk_management'])
                
                print(f"✅ Configuration loaded from {self.config_file}")
            else:
                print(f"⚠️ Config file {self.config_file} not found, using defaults")
                self.save_config()  # Create default config file
                
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("Using default configuration")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_dict = {
                'exchange': asdict(self.config.exchange),
                'backtest': asdict(self.config.backtest),
                'strategy': asdict(self.config.strategy),
                'risk_management': asdict(self.config.risk_management)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            print(f"✅ Configuration saved to {self.config_file}")
            
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def load_from_env(self):
        """Load sensitive data from environment variables"""
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load API credentials from environment
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')
        
        if api_key and api_secret:
            self.config.exchange.api_key = api_key
            self.config.exchange.api_secret = api_secret
            print("✅ API credentials loaded from environment")
        else:
            print("⚠️ API credentials not found in environment variables")
    
    def get_exchange_config(self) -> ExchangeConfig:
        """Get exchange configuration with credentials"""
        self.load_from_env()
        return self.config.exchange
    
    def update_strategy_params(self, **kwargs):
        """Update strategy parameters"""
        self.config.strategy.parameters.update(kwargs)
        self.save_config()
    
    def create_sample_config(self):
        """Create a sample configuration file"""
        sample_config = {
            "exchange": {
                "name": "kraken",
                "sandbox": False,
                "fee_rate": 0.001
            },
            "backtest": {
                "initial_balance": 10000.0,
                "fee_rate": 0.001,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "timeframe": "1h",
                "symbol": "BTC/USD"
            },
            "strategy": {
                "name": "sma_crossover",
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 20,
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70
                }
            },
            "risk_management": {
                "max_position_size": 0.1,
                "stop_loss_percent": 0.05,
                "take_profit_percent": 0.10,
                "max_drawdown_percent": 0.20
            }
        }
        
        with open('config_sample.json', 'w') as f:
            json.dump(sample_config, f, indent=4)
        
        print("✅ Sample configuration created: config_sample.json")

# Global config instance
config_manager = ConfigManager()

# Convenience functions
def get_config() -> TradingConfig:
    """Get the current trading configuration"""
    return config_manager.config

def get_exchange_config() -> ExchangeConfig:
    """Get exchange configuration with credentials loaded"""
    return config_manager.get_exchange_config()

def update_strategy_params(**kwargs):
    """Update strategy parameters"""
    config_manager.update_strategy_params(**kwargs)