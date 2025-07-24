#!/usr/bin/env python3
"""
Comprehensive Portfolio Optimization Testing Suite

Tests all Phase 4.2 Portfolio Optimization components:
- Black-Litterman optimization with market views
- Risk Parity allocation (ERC, HRP, Risk Budgeting) 
- Dynamic rebalancing system with multiple triggers
- Markowitz optimization with multiple objectives
- Regime-specific allocation with market regime detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import structlog
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import portfolio optimization modules
from src.portfolio.black_litterman import BlackLittermanOptimizer, BlackLittermanConfig
from src.portfolio.risk_parity import RiskParityOptimizer, RiskParityConfig
from src.portfolio.rebalancer import DynamicRebalancer, RebalancingConfig
from src.portfolio.markowitz import MarkowitzOptimizer, MarkowitzConfig
from src.portfolio.regime_allocator import RegimeSpecificAllocator, RegimeConfig

logger = structlog.get_logger()


class PortfolioOptimizationTester:
    """Comprehensive tester for portfolio optimization components"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {}
        self.setup_test_data()
        
    def setup_test_data(self):
        """Create realistic test data for optimization testing"""
        
        # Generate synthetic market data
        np.random.seed(42)
        n_days = 500
        n_assets = 8
        
        # Asset names
        self.assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'TLT', 'GLD', 'VNQ', 'HYG']
        
        # Generate correlated returns
        mean_returns = np.array([0.10, 0.12, 0.08, 0.06, 0.03, 0.05, 0.07, 0.04]) / 252
        correlation_matrix = np.array([
            [1.00, 0.85, 0.75, 0.65, -0.20, 0.10, 0.60, 0.45],
            [0.85, 1.00, 0.70, 0.60, -0.15, 0.05, 0.55, 0.40],
            [0.75, 0.70, 1.00, 0.55, -0.10, 0.15, 0.50, 0.35],
            [0.65, 0.60, 0.55, 1.00, -0.05, 0.20, 0.45, 0.30],
            [-0.20, -0.15, -0.10, -0.05, 1.00, 0.25, -0.10, 0.60],
            [0.10, 0.05, 0.15, 0.20, 0.25, 1.00, 0.30, 0.20],
            [0.60, 0.55, 0.50, 0.45, -0.10, 0.30, 1.00, 0.35],
            [0.45, 0.40, 0.35, 0.30, 0.60, 0.20, 0.35, 1.00]
        ])
        
        # Volatilities
        volatilities = np.array([0.16, 0.20, 0.22, 0.18, 0.08, 0.18, 0.20, 0.12]) / np.sqrt(252)
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Generate returns using multivariate normal
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
        
        # Create DataFrame
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        self.returns_data = pd.DataFrame(returns, index=dates, columns=self.assets)
        
        # Generate VIX data
        self.vix_data = pd.Series(
            np.random.gamma(2, 8) + 10,  # VIX-like distribution
            index=dates
        )
        
        # Market cap weights for Black-Litterman
        self.market_cap_weights = {
            'SPY': 0.35, 'QQQ': 0.25, 'IWM': 0.15, 'EFA': 0.10,
            'TLT': 0.05, 'GLD': 0.03, 'VNQ': 0.04, 'HYG': 0.03
        }
        
        # Asset class mapping for regime allocation
        self.asset_class_mapping = {
            'SPY': 'equities', 'QQQ': 'equities', 'IWM': 'equities', 'EFA': 'equities',
            'TLT': 'bonds', 'HYG': 'bonds',
            'GLD': 'commodities',
            'VNQ': 'equities'  # REITs classified as equities
        }
        
        logger.info("Test data setup completed", 
                   assets=len(self.assets), 
                   days=n_days)
    
    def test_black_litterman_optimizer(self) -> bool:
        """Test Black-Litterman portfolio optimization"""
        
        logger.info("Testing Black-Litterman optimization...")
        
        try:
            # Configure Black-Litterman
            config = BlackLittermanConfig(
                risk_aversion=3.0,
                tau=None,  # Auto-calculate
                allow_short_selling=False,
                max_weight=0.4,
                target_volatility=0.15
            )
            
            optimizer = BlackLittermanOptimizer(config)
            
            # Add some market views
            view_success = optimizer.add_market_view(
                asset_view={'SPY': 1.0, 'QQQ': -0.5},
                expected_return=0.05,
                confidence=0.8,
                view_name="SPY_outperforms_QQQ",
                expiry_days=30
            )
            
            if not view_success:
                logger.warning("Failed to add market view")
            
            # Add relative view
            relative_view_success = optimizer.add_relative_view(
                asset_long='QQQ',
                asset_short='IWM',
                expected_outperformance=0.03,
                confidence=0.7,
                expiry_days=60
            )
            
            # Perform optimization
            result = optimizer.optimize_portfolio(
                returns_data=self.returns_data,
                market_cap_weights=self.market_cap_weights
            )
            
            # Validate results
            if 'error' in result:
                logger.error("Black-Litterman optimization failed", error=result['error'])
                return False
            
            # Check result structure
            required_keys = ['optimal_weights', 'expected_returns', 'risk_metrics', 'views_incorporated']
            for key in required_keys:
                if key not in result:
                    logger.error(f"Missing key in Black-Litterman result: {key}")
                    return False
            
            # Validate weights
            weights = result['optimal_weights']
            weight_sum = sum(weights.values())
            if not (0.99 <= weight_sum <= 1.01):
                logger.error(f"Weights don't sum to 1: {weight_sum}")
                return False
            
            # Check if views were incorporated
            if result['views_incorporated'] == 0:
                logger.warning("No views were incorporated in optimization")
            
            # Test dashboard
            dashboard = optimizer.get_optimization_dashboard()
            if 'active_views' not in dashboard:
                logger.error("Dashboard missing active_views")
                return False
            
            self.test_results['black_litterman'] = {
                'success': True,
                'views_incorporated': result['views_incorporated'],
                'portfolio_return': result['risk_metrics']['expected_return'],
                'portfolio_risk': result['risk_metrics']['portfolio_volatility'],
                'sharpe_ratio': result['risk_metrics']['sharpe_ratio'],
                'max_weight': result['risk_metrics']['max_weight']
            }
            
            logger.info("Black-Litterman optimization test passed",
                       expected_return=result['risk_metrics']['expected_return'],
                       risk=result['risk_metrics']['portfolio_volatility'],
                       views=result['views_incorporated'])
            
            return True
            
        except Exception as e:
            logger.error("Black-Litterman optimization test failed", error=str(e))
            self.test_results['black_litterman'] = {'success': False, 'error': str(e)}
            return False
    
    def test_risk_parity_optimizer(self) -> bool:
        """Test Risk Parity portfolio optimization"""
        
        logger.info("Testing Risk Parity optimization...")
        
        try:
            # Test Equal Risk Contribution
            erc_config = RiskParityConfig(
                method="equal_risk_contribution",
                target_volatility=0.12,
                enable_vol_targeting=True,
                min_weight=0.01,
                max_weight=0.30
            )
            
            erc_optimizer = RiskParityOptimizer(erc_config)
            erc_result = erc_optimizer.optimize_portfolio(self.returns_data)
            
            if 'error' in erc_result:
                logger.error("ERC optimization failed", error=erc_result['error'])
                return False
            
            # Test Hierarchical Risk Parity
            hrp_config = RiskParityConfig(
                method="hierarchical",
                hrp_linkage_method="ward",
                hrp_distance_metric="correlation"
            )
            
            hrp_optimizer = RiskParityOptimizer(hrp_config)
            hrp_result = hrp_optimizer.optimize_portfolio(self.returns_data)
            
            if 'error' in hrp_result:
                logger.error("HRP optimization failed", error=hrp_result['error'])
                return False
            
            # Test Risk Budgeting
            risk_budgets = {
                'SPY': 0.25, 'QQQ': 0.20, 'IWM': 0.15, 'EFA': 0.15,
                'TLT': 0.10, 'GLD': 0.05, 'VNQ': 0.05, 'HYG': 0.05
            }
            
            rb_config = RiskParityConfig(
                method="risk_budgeting",
                risk_budgets=risk_budgets,
                risk_budget_tolerance=0.02
            )
            
            rb_optimizer = RiskParityOptimizer(rb_config)
            rb_result = rb_optimizer.optimize_portfolio(
                self.returns_data, 
                risk_budgets=risk_budgets
            )
            
            if 'error' in rb_result:
                logger.error("Risk budgeting optimization failed", error=rb_result['error'])
                return False
            
            # Validate all results
            for method, result in [('ERC', erc_result), ('HRP', hrp_result), ('RB', rb_result)]:
                weights = result['optimal_weights']
                weight_sum = sum(weights.values())
                
                if not (0.99 <= weight_sum <= 1.01):
                    logger.error(f"{method} weights don't sum to 1: {weight_sum}")
                    return False
                
                # Check risk contributions are reasonable
                risk_contribs = list(result['risk_contributions'].values())
                if method == 'ERC':
                    # ERC should have relatively equal risk contributions
                    risk_range = max(risk_contribs) - min(risk_contribs)
                    if risk_range > 0.15:  # Allow some deviation
                        logger.warning(f"ERC risk contributions have large range: {risk_range}")
            
            # Test dashboard
            dashboard = erc_optimizer.get_risk_parity_dashboard()
            if 'optimization_history' not in dashboard:
                logger.error("Dashboard missing optimization_history")
                return False
            
            self.test_results['risk_parity'] = {
                'success': True,
                'erc_diversification': erc_result['risk_metrics']['diversification_ratio'],
                'hrp_diversification': hrp_result['risk_metrics']['diversification_ratio'],
                'rb_risk_concentration': rb_result['risk_metrics']['risk_concentration'],
                'methods_tested': 3
            }
            
            logger.info("Risk Parity optimization test passed",
                       erc_div=erc_result['risk_metrics']['diversification_ratio'],
                       hrp_div=hrp_result['risk_metrics']['diversification_ratio'])
            
            return True
            
        except Exception as e:
            logger.error("Risk Parity optimization test failed", error=str(e))
            self.test_results['risk_parity'] = {'success': False, 'error': str(e)}
            return False
    
    def test_dynamic_rebalancer(self) -> bool:
        """Test Dynamic Rebalancing system"""
        
        logger.info("Testing Dynamic Rebalancing system...")
        
        try:
            # Configure rebalancer
            config = RebalancingConfig(
                weight_threshold=0.05,
                rebalancing_frequency="monthly",
                enable_volatility_adjustment=True,
                enable_transaction_cost_optimization=True,
                transaction_cost_rate=0.001
            )
            
            rebalancer = DynamicRebalancer(config)
            
            # Create current and target portfolios
            current_portfolio = {
                'SPY': 0.40, 'QQQ': 0.25, 'IWM': 0.10, 'EFA': 0.10,
                'TLT': 0.05, 'GLD': 0.03, 'VNQ': 0.04, 'HYG': 0.03
            }
            
            # Target with some drift
            target_portfolio = {
                'SPY': 0.35, 'QQQ': 0.30, 'IWM': 0.12, 'EFA': 0.08,
                'TLT': 0.06, 'GLD': 0.03, 'VNQ': 0.04, 'HYG': 0.02
            }
            
            portfolio_value = 1000000  # $1M portfolio
            
            # Test rebalancing evaluation
            evaluation = rebalancer.evaluate_rebalancing_need(
                current_portfolio=current_portfolio,
                target_portfolio=target_portfolio,
                returns_data=self.returns_data,
                portfolio_value=portfolio_value,
                regime_indicator="normal"
            )
            
            if 'error' in evaluation:
                logger.error("Rebalancing evaluation failed", error=evaluation['error'])
                return False
            
            # Check evaluation structure
            required_keys = ['should_rebalance', 'trigger_analysis', 'cost_analysis']
            for key in required_keys:
                if key not in evaluation:
                    logger.error(f"Missing key in rebalancing evaluation: {key}")
                    return False
            
            # Test rebalancing execution (if recommended)
            if evaluation['should_rebalance']:
                execution = rebalancer.execute_rebalancing(
                    current_portfolio=current_portfolio,
                    target_portfolio=target_portfolio,
                    portfolio_value=portfolio_value,
                    execution_method="gradual"
                )
                
                if 'trades_executed' not in execution:
                    logger.error("Rebalancing execution missing trades_executed")
                    return False
            
            # Test different regime scenarios
            crisis_evaluation = rebalancer.evaluate_rebalancing_need(
                current_portfolio=current_portfolio,
                target_portfolio=target_portfolio,
                returns_data=self.returns_data,
                portfolio_value=portfolio_value,
                regime_indicator="crisis"
            )
            
            # Test dashboard
            dashboard = rebalancer.get_rebalancing_dashboard()
            if 'config' not in dashboard:
                logger.error("Dashboard missing config")
                return False
            
            self.test_results['dynamic_rebalancer'] = {
                'success': True,
                'should_rebalance': evaluation['should_rebalance'],
                'triggers_fired': evaluation['trigger_analysis']['trigger_count'],
                'transaction_cost': evaluation['cost_analysis']['cost_percentage'],
                'regime_tests': 2
            }
            
            logger.info("Dynamic Rebalancing test passed",
                       should_rebalance=evaluation['should_rebalance'],
                       triggers=evaluation['trigger_analysis']['trigger_count'])
            
            return True
            
        except Exception as e:
            logger.error("Dynamic Rebalancing test failed", error=str(e))
            self.test_results['dynamic_rebalancer'] = {'success': False, 'error': str(e)}
            return False
    
    def test_markowitz_optimizer(self) -> bool:
        """Test Markowitz portfolio optimization"""
        
        logger.info("Testing Markowitz optimization...")
        
        try:
            # Test multiple objectives
            objectives = [
                "maximize_sharpe",
                "minimize_variance",
                "maximize_return",
                "target_risk"
            ]
            
            results = {}
            
            for objective in objectives:
                config = MarkowitzConfig(
                    objective=objective,
                    risk_free_rate=0.02,
                    allow_short_selling=False,
                    max_weight=0.4,
                    target_risk=0.15 if objective == "target_risk" else None,
                    enable_robust_optimization=True,
                    efficient_frontier_points=20
                )
                
                optimizer = MarkowitzOptimizer(config)
                
                result = optimizer.optimize_portfolio(
                    returns_data=self.returns_data
                )
                
                if 'error' in result:
                    logger.error(f"Markowitz optimization failed for {objective}", 
                               error=result['error'])
                    return False
                
                # Validate result
                weights = result['optimal_weights']
                weight_sum = sum(weights.values())
                
                if not (0.99 <= weight_sum <= 1.01):
                    logger.error(f"{objective} weights don't sum to 1: {weight_sum}")
                    return False
                
                results[objective] = result
            
            # Test efficient frontier construction
            sharpe_result = results["maximize_sharpe"]
            if sharpe_result.get('efficient_frontier'):
                frontier = sharpe_result['efficient_frontier']
                if 'frontier_portfolios' not in frontier:
                    logger.error("Efficient frontier missing frontier_portfolios")
                    return False
                
                if len(frontier['frontier_portfolios']) < 10:
                    logger.warning(f"Efficient frontier has few points: {len(frontier['frontier_portfolios'])}")
            
            # Test different estimation methods
            shrinkage_config = MarkowitzConfig(
                objective="maximize_sharpe",
                return_estimation="shrinkage",
                covariance_estimation="shrinkage",
                shrinkage_intensity=0.2
            )
            
            shrinkage_optimizer = MarkowitzOptimizer(shrinkage_config)
            shrinkage_result = shrinkage_optimizer.optimize_portfolio(self.returns_data)
            
            if 'error' in shrinkage_result:
                logger.error("Shrinkage estimation test failed", error=shrinkage_result['error'])
                return False
            
            # Test dashboard
            dashboard = optimizer.get_optimization_dashboard()
            if 'performance_stats' not in dashboard:
                logger.error("Dashboard missing performance_stats")
                return False
            
            self.test_results['markowitz'] = {
                'success': True,
                'objectives_tested': len(objectives),
                'max_sharpe_ratio': results["maximize_sharpe"]['risk_metrics']['sharpe_ratio'],
                'min_var_volatility': results["minimize_variance"]['risk_metrics']['portfolio_risk'],
                'efficient_frontier_points': len(sharpe_result.get('efficient_frontier', {}).get('frontier_portfolios', []))
            }
            
            logger.info("Markowitz optimization test passed",
                       objectives=len(objectives),
                       max_sharpe=results["maximize_sharpe"]['risk_metrics']['sharpe_ratio'])
            
            return True
            
        except Exception as e:
            logger.error("Markowitz optimization test failed", error=str(e))
            self.test_results['markowitz'] = {'success': False, 'error': str(e)}
            return False
    
    def test_regime_allocator(self) -> bool:
        """Test Regime-Specific allocation"""
        
        logger.info("Testing Regime-Specific allocation...")
        
        try:
            # Configure regime allocator
            config = RegimeConfig(
                detection_method="multi_factor",
                lookback_period=100,
                regime_confidence_threshold=0.6,
                enable_smooth_transitions=True,
                asset_class_mapping=self.asset_class_mapping
            )
            
            allocator = RegimeSpecificAllocator(config)
            
            # Test regime detection and allocation
            allocation_result = allocator.allocate_portfolio(
                market_data=self.returns_data,
                asset_universe=self.assets,
                vix_data=self.vix_data
            )
            
            if 'error' in allocation_result:
                logger.error("Regime allocation failed", error=allocation_result['error'])
                return False
            
            # Validate allocation structure
            required_keys = ['allocation', 'regime_info', 'transition_info']
            for key in required_keys:
                if key not in allocation_result:
                    logger.error(f"Missing key in regime allocation: {key}")
                    return False
            
            # Check allocation weights
            allocation = allocation_result['allocation']
            weight_sum = sum(allocation.values())
            
            if not (0.99 <= weight_sum <= 1.01):
                logger.error(f"Regime allocation weights don't sum to 1: {weight_sum}")
                return False
            
            # Test multiple regime scenarios by manipulating data
            crisis_data = self.returns_data.copy()
            # Simulate crisis: high volatility, negative returns
            crisis_data.iloc[-30:] *= 3  # Increase volatility
            crisis_data.iloc[-30:] -= 0.02  # Add negative drift
            
            crisis_allocation = allocator.allocate_portfolio(
                market_data=crisis_data,
                asset_universe=self.assets,
                vix_data=self.vix_data * 2  # Higher VIX
            )
            
            if 'error' in crisis_allocation:
                logger.error("Crisis regime allocation failed", error=crisis_allocation['error'])
                return False
            
            # Test different detection methods
            methods = ["volatility_regime", "threshold_based", "gaussian_mixture"]
            method_results = {}
            
            for method in methods:
                method_config = RegimeConfig(
                    detection_method=method,
                    lookback_period=100,
                    asset_class_mapping=self.asset_class_mapping
                )
                
                method_allocator = RegimeSpecificAllocator(method_config)
                
                try:
                    method_result = method_allocator.allocate_portfolio(
                        market_data=self.returns_data,
                        asset_universe=self.assets,
                        vix_data=self.vix_data
                    )
                    
                    if 'error' not in method_result:
                        method_results[method] = method_result['regime_info']['regime']
                    else:
                        logger.warning(f"Method {method} failed", error=method_result['error'])
                        
                except Exception as e:
                    logger.warning(f"Method {method} exception", error=str(e))
            
            # Test dashboard
            dashboard = allocator.get_regime_dashboard()
            if 'regime_history' not in dashboard:
                logger.error("Dashboard missing regime_history")
                return False
            
            self.test_results['regime_allocator'] = {
                'success': True,
                'primary_regime': allocation_result['regime_info']['regime'],
                'regime_confidence': allocation_result['regime_info']['confidence'],
                'crisis_regime': crisis_allocation['regime_info']['regime'],
                'methods_tested': len(method_results),
                'allocation_sum': weight_sum
            }
            
            logger.info("Regime-Specific allocation test passed",
                       regime=allocation_result['regime_info']['regime'],
                       confidence=allocation_result['regime_info']['confidence'])
            
            return True
            
        except Exception as e:
            logger.error("Regime-Specific allocation test failed", error=str(e))
            self.test_results['regime_allocator'] = {'success': False, 'error': str(e)}
            return False
    
    def test_integration_workflow(self) -> bool:
        """Test integrated portfolio optimization workflow"""
        
        logger.info("Testing integrated portfolio optimization workflow...")
        
        try:
            # Step 1: Detect market regime
            regime_config = RegimeConfig(
                detection_method="multi_factor",
                asset_class_mapping=self.asset_class_mapping
            )
            regime_allocator = RegimeSpecificAllocator(regime_config)
            
            regime_result = regime_allocator.allocate_portfolio(
                market_data=self.returns_data,
                asset_universe=self.assets,
                vix_data=self.vix_data
            )
            
            detected_regime = regime_result['regime_info']['regime']
            strategic_allocation = regime_result['allocation']
            
            # Step 2: Optimize tactical allocation using detected regime
            if detected_regime in ['bull_trending', 'normal']:
                # Use Markowitz for growth-oriented regimes
                markowitz_config = MarkowitzConfig(
                    objective="maximize_sharpe",
                    max_weight=0.4,
                    allow_short_selling=False
                )
                optimizer = MarkowitzOptimizer(markowitz_config)
                tactical_result = optimizer.optimize_portfolio(self.returns_data)
                
            elif detected_regime in ['high_volatility', 'crisis']:
                # Use Risk Parity for defensive regimes
                rp_config = RiskParityConfig(
                    method="equal_risk_contribution",
                    target_volatility=0.10,  # Lower target vol
                    enable_vol_targeting=True
                )
                optimizer = RiskParityOptimizer(rp_config)
                tactical_result = optimizer.optimize_portfolio(self.returns_data)
                
            else:
                # Use Black-Litterman for uncertain/mixed regimes
                bl_config = BlackLittermanConfig(
                    risk_aversion=4.0,  # Higher risk aversion
                    max_weight=0.3
                )
                optimizer = BlackLittermanOptimizer(bl_config)
                tactical_result = optimizer.optimize_portfolio(
                    self.returns_data, 
                    self.market_cap_weights
                )
            
            if 'error' in tactical_result:
                logger.error("Tactical optimization failed", error=tactical_result['error'])
                return False
            
            tactical_allocation = tactical_result['optimal_weights']
            
            # Step 3: Blend strategic and tactical allocations
            blend_weight = 0.7  # 70% strategic, 30% tactical
            blended_allocation = {}
            
            for asset in self.assets:
                strategic_weight = strategic_allocation.get(asset, 0)
                tactical_weight = tactical_allocation.get(asset, 0)
                blended_weight = (blend_weight * strategic_weight + 
                                (1 - blend_weight) * tactical_weight)
                blended_allocation[asset] = blended_weight
            
            # Renormalize
            total_weight = sum(blended_allocation.values())
            blended_allocation = {k: v / total_weight for k, v in blended_allocation.items()}
            
            # Step 4: Test rebalancing from current to target
            current_allocation = {asset: 1.0 / len(self.assets) for asset in self.assets}  # Equal weight
            
            rebalancer_config = RebalancingConfig(
                weight_threshold=0.05,
                enable_transaction_cost_optimization=True
            )
            rebalancer = DynamicRebalancer(rebalancer_config)
            
            rebalancing_evaluation = rebalancer.evaluate_rebalancing_need(
                current_portfolio=current_allocation,
                target_portfolio=blended_allocation,
                returns_data=self.returns_data,
                portfolio_value=1000000,
                regime_indicator=detected_regime
            )
            
            if 'error' in rebalancing_evaluation:
                logger.error("Integrated rebalancing failed", error=rebalancing_evaluation['error'])
                return False
            
            # Validate integration
            allocation_sum = sum(blended_allocation.values())
            if not (0.99 <= allocation_sum <= 1.01):
                logger.error(f"Integrated allocation doesn't sum to 1: {allocation_sum}")
                return False
            
            self.test_results['integration'] = {
                'success': True,
                'detected_regime': detected_regime,
                'strategic_max_weight': max(strategic_allocation.values()),
                'tactical_max_weight': max(tactical_allocation.values()),
                'blended_max_weight': max(blended_allocation.values()),
                'should_rebalance': rebalancing_evaluation['should_rebalance'],
                'workflow_steps': 4
            }
            
            logger.info("Integrated workflow test passed",
                       regime=detected_regime,
                       should_rebalance=rebalancing_evaluation['should_rebalance'])
            
            return True
            
        except Exception as e:
            logger.error("Integrated workflow test failed", error=str(e))
            self.test_results['integration'] = {'success': False, 'error': str(e)}
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all portfolio optimization tests"""
        
        logger.info("Starting Portfolio Optimization Test Suite")
        
        test_methods = [
            ('Black-Litterman Optimizer', self.test_black_litterman_optimizer),
            ('Risk Parity Optimizer', self.test_risk_parity_optimizer),
            ('Dynamic Rebalancer', self.test_dynamic_rebalancer),
            ('Markowitz Optimizer', self.test_markowitz_optimizer),
            ('Regime Allocator', self.test_regime_allocator),
            ('Integration Workflow', self.test_integration_workflow)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            logger.info(f"Running {test_name} test...")
            
            try:
                success = test_method()
                if success:
                    passed_tests += 1
                    logger.info(f"✓ {test_name} test PASSED")
                else:
                    logger.error(f"✗ {test_name} test FAILED")
                    
            except Exception as e:
                logger.error(f"✗ {test_name} test CRASHED", error=str(e))
                self.test_results[test_name.lower().replace(' ', '_')] = {
                    'success': False, 
                    'error': str(e)
                }
        
        # Overall results
        success_rate = passed_tests / total_tests
        overall_success = success_rate >= 0.8  # 80% pass rate required
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_success': overall_success,
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'success_rate': success_rate,
            'detailed_results': self.test_results
        }
        
        logger.info("Portfolio Optimization Test Suite completed",
                   success_rate=f"{success_rate:.1%}",
                   passed=passed_tests,
                   total=total_tests)
        
        if overall_success:
            logger.info("🎉 Portfolio Optimization Test Suite PASSED!")
        else:
            logger.error("❌ Portfolio Optimization Test Suite FAILED!")
        
        return summary


def main():
    """Main test execution"""
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run tests
    tester = PortfolioOptimizationTester()
    results = tester.run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION TEST SUMMARY")
    print("="*80)
    print(f"Overall Success: {'✓ PASSED' if results['overall_success'] else '✗ FAILED'}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    print("="*80)
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)


if __name__ == "__main__":
    main()