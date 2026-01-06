"""
Advanced Analytics Dashboard for Phase 3
Comprehensive analytics and visualization for multi-modal assessment data
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import math
import uuid

# Import assessment components
from computer_vision import VisualAnalysisResult
from voice_analysis import VoiceAnalysisResult, SpeechEmotion
from gesture_recognition import HandTrackingResult, GestureType
from emotional_intelligence import EmotionalIntelligenceResult
from conversational_ai import ConversationalResponse, DialogueState
from multimodal_fusion import FusionResult, ModalityType
from learning_recommendations import LearningProfile, RecommendationsResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    PERFORMANCE = "performance"
    ENGAGEMENT = "engagement"
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    BEHAVIORAL = "behavioral"
    LEARNING = "learning"

class TimeRange(Enum):
    REAL_TIME = "real_time"
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"

class VisualizationType(Enum):
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter_plot"
    GAUGE = "gauge"
    PROGRESS_BAR = "progress_bar"

@dataclass
class MetricData:
    """Individual metric data point"""
    metric_id: str
    timestamp: float
    metric_type: MetricType
    value: float
    unit: str
    context: Dict[str, Any]

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    visualization_type: VisualizationType
    metrics: List[str]
    time_range: TimeRange
    position: Dict[str, int]
    size: Dict[str, int]
    refresh_interval: int

@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    report_id: str
    timestamp: float
    user_id: str
    session_id: str
    summary: Dict[str, Any]
    detailed_metrics: List[MetricData]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]

class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard for multi-modal assessment"""
    
    def __init__(self):
        self.metrics_history = {}
        self.dashboard_configs = {}
        self.reports = {}
        self.real_time_data = {}
        self.alerts = {}
        
        # Metric thresholds
        self.metric_thresholds = {
            'engagement_score': {'low': 0.3, 'high': 0.8},
            'cognitive_load': {'low': 0.2, 'high': 0.7},
            'stress_level': {'low': 0.2, 'high': 0.6},
            'accuracy': {'low': 0.4, 'high': 0.8},
            'confidence': {'low': 0.3, 'high': 0.8}
        }
        
        logger.info("Advanced Analytics Dashboard initialized")
    
    def add_metric_data(self, user_id: str, metric_data: MetricData):
        """Add metric data to analytics"""
        try:
            if user_id not in self.metrics_history:
                self.metrics_history[user_id] = deque(maxlen=1000)
            
            self.metrics_history[user_id].append(metric_data)
            
            # Check for alerts
            self._check_metric_alerts(user_id, metric_data)
            
            logger.debug(f"Added metric {metric_data.metric_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Add metric data error: {e}")
    
    def process_assessment_data(self, user_id: str, session_id: str, 
                               assessment_data: Dict[str, Any]):
        """Process multi-modal assessment data"""
        try:
            timestamp = time.time()
            
            # Process visual data
            if 'visual' in assessment_data:
                self._process_visual_metrics(user_id, timestamp, assessment_data['visual'])
            
            # Process voice data
            if 'voice' in assessment_data:
                self._process_voice_metrics(user_id, timestamp, assessment_data['voice'])
            
            # Process gesture data
            if 'gesture' in assessment_data:
                self._process_gesture_metrics(user_id, timestamp, assessment_data['gesture'])
            
            # Process emotional data
            if 'emotional' in assessment_data:
                self._process_emotional_metrics(user_id, timestamp, assessment_data['emotional'])
            
            # Process conversational data
            if 'conversational' in assessment_data:
                self._process_conversational_metrics(user_id, timestamp, assessment_data['conversational'])
            
            # Process fusion data
            if 'fusion' in assessment_data:
                self._process_fusion_metrics(user_id, timestamp, assessment_data['fusion'])
            
            # Process learning data
            if 'learning' in assessment_data:
                self._process_learning_metrics(user_id, timestamp, assessment_data['learning'])
            
            logger.info(f"Processed assessment data for user {user_id}, session {session_id}")
            
        except Exception as e:
            logger.error(f"Process assessment data error: {e}")
    
    def create_dashboard_config(self, user_id: str, dashboard_type: str = "default") -> Dict[str, Any]:
        """Create dashboard configuration"""
        try:
            config_id = str(uuid.uuid4())
            
            if dashboard_type == "default":
                widgets = self._create_default_widgets()
            elif dashboard_type == "learning":
                widgets = self._create_learning_widgets()
            elif dashboard_type == "performance":
                widgets = self._create_performance_widgets()
            else:
                widgets = self._create_custom_widgets(dashboard_type)
            
            config = {
                'config_id': config_id,
                'user_id': user_id,
                'dashboard_type': dashboard_type,
                'widgets': widgets,
                'layout': self._create_layout(),
                'theme': 'modern',
                'created_at': time.time()
            }
            
            self.dashboard_configs[user_id] = config
            logger.info(f"Created dashboard config for user {user_id}")
            return config
            
        except Exception as e:
            logger.error(f"Create dashboard config error: {e}")
            return {}
    
    def generate_analytics_report(self, user_id: str, session_id: str, 
                                 time_range: TimeRange = TimeRange.LAST_DAY) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        try:
            report_id = str(uuid.uuid4())
            timestamp = time.time()
            
            # Get metrics for time range
            metrics = self._get_metrics_by_time_range(user_id, time_range)
            
            # Generate summary
            summary = self._generate_summary_metrics(metrics)
            
            # Create visualizations
            visualizations = self._create_visualizations(metrics)
            
            # Generate insights
            insights = self._generate_insights(user_id, metrics)
            
            # Generate recommendations
            recommendations = self._generate_analytics_recommendations(user_id, metrics)
            
            report = AnalyticsReport(
                report_id=report_id,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                summary=summary,
                detailed_metrics=metrics,
                visualizations=visualizations,
                insights=insights,
                recommendations=recommendations
            )
            
            self.reports[report_id] = report
            logger.info(f"Generated analytics report {report_id} for user {user_id}")
            return report
            
        except Exception as e:
            logger.error(f"Generate analytics report error: {e}")
            return self._create_default_report(user_id, session_id)
    
    def get_real_time_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            if user_id not in self.metrics_history:
                return {"error": "No data available"}
            
            recent_metrics = list(self.metrics_history[user_id])[-50:]  # Last 50 metrics
            
            dashboard_data = {
                'timestamp': time.time(),
                'user_id': user_id,
                'real_time_metrics': self._extract_real_time_metrics(recent_metrics),
                'alerts': self.alerts.get(user_id, []),
                'trends': self._calculate_real_time_trends(recent_metrics),
                'status': self._get_overall_status(recent_metrics)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Get real-time dashboard error: {e}")
            return {"error": str(e)}
    
    def _process_visual_metrics(self, user_id: str, timestamp: float, visual_data: Dict[str, Any]):
        """Process visual analysis metrics"""
        try:
            metrics = [
                MetricData(str(uuid.uuid4()), timestamp, MetricType.ENGAGEMENT, 
                          visual_data.get('attention_score', 0), "score", {'source': 'visual'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.BEHAVIORAL, 
                          visual_data.get('eye_contact_consistency', 0), "score", {'source': 'visual'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.PERFORMANCE, 
                          visual_data.get('posture_score', 0), "score", {'source': 'visual'})
            ]
            
            for metric in metrics:
                self.add_metric_data(user_id, metric)
                
        except Exception as e:
            logger.error(f"Process visual metrics error: {e}")
    
    def _process_voice_metrics(self, user_id: str, timestamp: float, voice_data: Dict[str, Any]):
        """Process voice analysis metrics"""
        try:
            metrics = [
                MetricData(str(uuid.uuid4()), timestamp, MetricType.ENGAGEMENT, 
                          voice_data.get('speaking_rate', 0), "rate", {'source': 'voice'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.EMOTIONAL, 
                          voice_data.get('pitch_mean', 0), "hz", {'source': 'voice'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.COGNITIVE, 
                          voice_data.get('energy_mean', 0), "energy", {'source': 'voice'})
            ]
            
            for metric in metrics:
                self.add_metric_data(user_id, metric)
                
        except Exception as e:
            logger.error(f"Process voice metrics error: {e}")
    
    def _process_gesture_metrics(self, user_id: str, timestamp: float, gesture_data: Dict[str, Any]):
        """Process gesture recognition metrics"""
        try:
            metrics = [
                MetricData(str(uuid.uuid4()), timestamp, MetricType.BEHAVIORAL, 
                          len(gesture_data.get('gesture_results', [])), "count", {'source': 'gesture'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.ENGAGEMENT, 
                          gesture_data.get('activity_level', 0), "level", {'source': 'gesture'})
            ]
            
            for metric in metrics:
                self.add_metric_data(user_id, metric)
                
        except Exception as e:
            logger.error(f"Process gesture metrics error: {e}")
    
    def _process_emotional_metrics(self, user_id: str, timestamp: float, emotional_data: Dict[str, Any]):
        """Process emotional intelligence metrics"""
        try:
            metrics = [
                MetricData(str(uuid.uuid4()), timestamp, MetricType.EMOTIONAL, 
                          emotional_data.get('confidence_level', 0), "score", {'source': 'emotional'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.COGNITIVE, 
                          emotional_data.get('stress_level', 0), "level", {'source': 'emotional'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.ENGAGEMENT, 
                          emotional_data.get('engagement_score', 0), "score", {'source': 'emotional'})
            ]
            
            for metric in metrics:
                self.add_metric_data(user_id, metric)
                
        except Exception as e:
            logger.error(f"Process emotional metrics error: {e}")
    
    def _process_conversational_metrics(self, user_id: str, timestamp: float, conversational_data: Dict[str, Any]):
        """Process conversational AI metrics"""
        try:
            metrics = [
                MetricData(str(uuid.uuid4()), timestamp, MetricType.LEARNING, 
                          conversational_data.get('dialogue_quality', 0), "score", {'source': 'conversational'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.ENGAGEMENT, 
                          conversational_data.get('response_time', 0), "seconds", {'source': 'conversational'})
            ]
            
            for metric in metrics:
                self.add_metric_data(user_id, metric)
                
        except Exception as e:
            logger.error(f"Process conversational metrics error: {e}")
    
    def _process_fusion_metrics(self, user_id: str, timestamp: float, fusion_data: Dict[str, Any]):
        """Process multi-modal fusion metrics"""
        try:
            metrics = [
                MetricData(str(uuid.uuid4()), timestamp, MetricType.COGNITIVE, 
                          fusion_data.get('cognitive_load', 0), "load", {'source': 'fusion'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.ENGAGEMENT, 
                          fusion_data.get('engagement_score', 0), "score", {'source': 'fusion'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.LEARNING, 
                          fusion_data.get('learning_readiness', 0), "readiness", {'source': 'fusion'})
            ]
            
            for metric in metrics:
                self.add_metric_data(user_id, metric)
                
        except Exception as e:
            logger.error(f"Process fusion metrics error: {e}")
    
    def _process_learning_metrics(self, user_id: str, timestamp: float, learning_data: Dict[str, Any]):
        """Process learning recommendations metrics"""
        try:
            metrics = [
                MetricData(str(uuid.uuid4()), timestamp, MetricType.LEARNING, 
                          learning_data.get('accuracy', 0), "score", {'source': 'learning'}),
                MetricData(str(uuid.uuid4()), timestamp, MetricType.PERFORMANCE, 
                          learning_data.get('completion_rate', 0), "rate", {'source': 'learning'})
            ]
            
            for metric in metrics:
                self.add_metric_data(user_id, metric)
                
        except Exception as e:
            logger.error(f"Process learning metrics error: {e}")
    
    def _create_default_widgets(self) -> List[Dict[str, Any]]:
        """Create default dashboard widgets"""
        return [
            {
                'widget_id': 'engagement_gauge',
                'title': 'Engagement Level',
                'type': VisualizationType.GAUGE.value,
                'metrics': ['engagement_score'],
                'position': {'x': 0, 'y': 0},
                'size': {'width': 2, 'height': 2}
            },
            {
                'widget_id': 'cognitive_load_chart',
                'title': 'Cognitive Load Trend',
                'type': VisualizationType.LINE_CHART.value,
                'metrics': ['cognitive_load'],
                'position': {'x': 2, 'y': 0},
                'size': {'width': 4, 'height': 2}
            },
            {
                'widget_id': 'performance_pie',
                'title': 'Performance Distribution',
                'type': VisualizationType.PIE_CHART.value,
                'metrics': ['accuracy', 'completion_rate'],
                'position': {'x': 6, 'y': 0},
                'size': {'width': 2, 'height': 2}
            }
        ]
    
    def _create_learning_widgets(self) -> List[Dict[str, Any]]:
        """Create learning-focused widgets"""
        return [
            {
                'widget_id': 'learning_progress',
                'title': 'Learning Progress',
                'type': VisualizationType.PROGRESS_BAR.value,
                'metrics': ['accuracy'],
                'position': {'x': 0, 'y': 0},
                'size': {'width': 6, 'height': 1}
            },
            {
                'widget_id': 'skill_breakdown',
                'title': 'Skill Breakdown',
                'type': VisualizationType.BAR_CHART.value,
                'metrics': ['skill_scores'],
                'position': {'x': 0, 'y': 1},
                'size': {'width': 6, 'height': 3}
            }
        ]
    
    def _create_performance_widgets(self) -> List[Dict[str, Any]]:
        """Create performance-focused widgets"""
        return [
            {
                'widget_id': 'performance_metrics',
                'title': 'Performance Metrics',
                'type': VisualizationType.BAR_CHART.value,
                'metrics': ['accuracy', 'speed', 'efficiency'],
                'position': {'x': 0, 'y': 0},
                'size': {'width': 8, 'height': 4}
            }
        ]
    
    def _create_custom_widgets(self, dashboard_type: str) -> List[Dict[str, Any]]:
        """Create custom widgets based on type"""
        return self._create_default_widgets()
    
    def _create_layout(self) -> Dict[str, Any]:
        """Create dashboard layout"""
        return {
            'columns': 8,
            'row_height': 100,
            'margin': {'x': 10, 'y': 10},
            'padding': {'x': 5, 'y': 5}
        }
    
    def _get_metrics_by_time_range(self, user_id: str, time_range: TimeRange) -> List[MetricData]:
        """Get metrics within specified time range"""
        try:
            if user_id not in self.metrics_history:
                return []
            
            current_time = time.time()
            metrics = list(self.metrics_history[user_id])
            
            if time_range == TimeRange.REAL_TIME:
                cutoff = current_time - 300  # Last 5 minutes
            elif time_range == TimeRange.LAST_HOUR:
                cutoff = current_time - 3600
            elif time_range == TimeRange.LAST_DAY:
                cutoff = current_time - 86400
            elif time_range == TimeRange.LAST_WEEK:
                cutoff = current_time - 604800
            elif time_range == TimeRange.LAST_MONTH:
                cutoff = current_time - 2592000
            else:
                cutoff = 0
            
            return [m for m in metrics if m.timestamp >= cutoff]
            
        except Exception as e:
            logger.error(f"Get metrics by time range error: {e}")
            return []
    
    def _generate_summary_metrics(self, metrics: List[MetricData]) -> Dict[str, Any]:
        """Generate summary statistics"""
        try:
            if not metrics:
                return {}
            
            summary = {}
            
            # Group by metric type
            by_type = {}
            for metric in metrics:
                mtype = metric.metric_type.value
                if mtype not in by_type:
                    by_type[mtype] = []
                by_type[mtype].append(metric.value)
            
            # Calculate statistics for each type
            for mtype, values in by_type.items():
                if values:
                    summary[mtype] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Generate summary metrics error: {e}")
            return {}
    
    def _create_visualizations(self, metrics: List[MetricData]) -> List[Dict[str, Any]]:
        """Create visualization configurations"""
        try:
            visualizations = []
            
            # Time series chart for engagement
            engagement_metrics = [m for m in metrics if 'engagement' in str(m.context.get('source', ''))]
            if engagement_metrics:
                visualizations.append({
                    'type': 'line_chart',
                    'title': 'Engagement Over Time',
                    'data': [{'time': m.timestamp, 'value': m.value} for m in engagement_metrics],
                    'config': {'x_axis': 'time', 'y_axis': 'engagement_score'}
                })
            
            # Pie chart for metric distribution
            metric_counts = {}
            for metric in metrics:
                source = metric.context.get('source', 'unknown')
                metric_counts[source] = metric_counts.get(source, 0) + 1
            
            if metric_counts:
                visualizations.append({
                    'type': 'pie_chart',
                    'title': 'Data Source Distribution',
                    'data': [{'source': k, 'count': v} for k, v in metric_counts.items()],
                    'config': {'labels': 'source', 'values': 'count'}
                })
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Create visualizations error: {e}")
            return []
    
    def _generate_insights(self, user_id: str, metrics: List[MetricData]) -> List[str]:
        """Generate insights from metrics"""
        try:
            insights = []
            
            if not metrics:
                return ["No data available for insights"]
            
            # Engagement insights
            engagement_metrics = [m for m in metrics if 'engagement' in str(m.context.get('source', ''))]
            if engagement_metrics:
                avg_engagement = sum(m.value for m in engagement_metrics) / len(engagement_metrics)
                if avg_engagement > 0.8:
                    insights.append("High engagement levels detected - user is highly motivated")
                elif avg_engagement < 0.4:
                    insights.append("Low engagement detected - consider intervention strategies")
            
            # Cognitive load insights
            cognitive_metrics = [m for m in metrics if 'cognitive' in str(m.context.get('source', ''))]
            if cognitive_metrics:
                avg_load = sum(m.value for m in cognitive_metrics) / len(cognitive_metrics)
                if avg_load > 0.7:
                    insights.append("High cognitive load - user may need breaks or simplified content")
            
            return insights
            
        except Exception as e:
            logger.error(f"Generate insights error: {e}")
            return []
    
    def _generate_analytics_recommendations(self, user_id: str, metrics: List[MetricData]) -> List[str]:
        """Generate recommendations based on analytics"""
        try:
            recommendations = []
            
            if not metrics:
                return ["No data available for recommendations"]
            
            # Performance recommendations
            performance_metrics = [m for m in metrics if m.metric_type == MetricType.PERFORMANCE]
            if performance_metrics:
                avg_performance = sum(m.value for m in performance_metrics) / len(performance_metrics)
                if avg_performance < 0.5:
                    recommendations.append("Consider providing additional support and resources")
            
            # Engagement recommendations
            engagement_metrics = [m for m in metrics if m.metric_type == MetricType.ENGAGEMENT]
            if engagement_metrics:
                avg_engagement = sum(m.value for m in engagement_metrics) / len(engagement_metrics)
                if avg_engagement < 0.6:
                    recommendations.append("Implement interactive elements to increase engagement")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Generate analytics recommendations error: {e}")
            return []
    
    def _check_metric_alerts(self, user_id: str, metric_data: MetricData):
        """Check for metric alerts"""
        try:
            metric_name = metric_data.context.get('source', 'unknown')
            value = metric_data.value
            
            if metric_name in self.metric_thresholds:
                thresholds = self.metric_thresholds[metric_name]
                
                if value < thresholds['low']:
                    self._create_alert(user_id, 'low', metric_name, value)
                elif value > thresholds['high']:
                    self._create_alert(user_id, 'high', metric_name, value)
                    
        except Exception as e:
            logger.error(f"Check metric alerts error: {e}")
    
    def _create_alert(self, user_id: str, alert_type: str, metric_name: str, value: float):
        """Create alert for metric threshold breach"""
        try:
            if user_id not in self.alerts:
                self.alerts[user_id] = deque(maxlen=100)
            
            alert = {
                'alert_id': str(uuid.uuid4()),
                'timestamp': time.time(),
                'type': alert_type,
                'metric': metric_name,
                'value': value,
                'message': f"{alert_type.title()} {metric_name} detected: {value:.2f}"
            }
            
            self.alerts[user_id].append(alert)
            logger.warning(f"Alert created for user {user_id}: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Create alert error: {e}")
    
    def _extract_real_time_metrics(self, metrics: List[MetricData]) -> Dict[str, Any]:
        """Extract real-time metrics from recent data"""
        try:
            real_time = {}
            
            for metric in metrics[-10:]:  # Last 10 metrics
                source = metric.context.get('source', 'unknown')
                real_time[source] = metric.value
            
            return real_time
            
        except Exception as e:
            logger.error(f"Extract real-time metrics error: {e}")
            return {}
    
    def _calculate_real_time_trends(self, metrics: List[MetricData]) -> Dict[str, str]:
        """Calculate real-time trends"""
        try:
            trends = {}
            
            if len(metrics) < 2:
                return trends
            
            # Simple trend calculation
            recent = metrics[-5:]
            older = metrics[-10:-5]
            
            if recent and older:
                recent_avg = sum(m.value for m in recent) / len(recent)
                older_avg = sum(m.value for m in older) / len(older)
                
                if recent_avg > older_avg * 1.1:
                    trends['overall'] = 'increasing'
                elif recent_avg < older_avg * 0.9:
                    trends['overall'] = 'decreasing'
                else:
                    trends['overall'] = 'stable'
            
            return trends
            
        except Exception as e:
            logger.error(f"Calculate real-time trends error: {e}")
            return {}
    
    def _get_overall_status(self, metrics: List[MetricData]) -> str:
        """Get overall system status"""
        try:
            if not metrics:
                return 'no_data'
            
            # Check for critical alerts
            recent_metrics = metrics[-20:]
            
            for metric in recent_metrics:
                if metric.value < 0.2:  # Very low values
                    return 'critical'
            
            # Check for warnings
            low_count = sum(1 for m in recent_metrics if m.value < 0.4)
            if low_count > len(recent_metrics) * 0.3:
                return 'warning'
            
            # Check for good status
            high_count = sum(1 for m in recent_metrics if m.value > 0.7)
            if high_count > len(recent_metrics) * 0.6:
                return 'excellent'
            
            return 'good'
            
        except Exception as e:
            logger.error(f"Get overall status error: {e}")
            return 'unknown'
    
    def _create_default_report(self, user_id: str, session_id: str) -> AnalyticsReport:
        """Create default analytics report"""
        return AnalyticsReport(
            report_id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            summary={},
            detailed_metrics=[],
            visualizations=[],
            insights=["No data available"],
            recommendations=["Collect more assessment data"]
        )
    
    def get_dashboard_summary(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard summary"""
        try:
            if user_id not in self.metrics_history:
                return {"error": "No data available"}
            
            metrics = list(self.metrics_history[user_id])
            alerts = list(self.alerts.get(user_id, []))
            
            summary = {
                'total_metrics': len(metrics),
                'active_alerts': len([a for a in alerts if a['timestamp'] > time.time() - 3600]),
                'data_sources': list(set(m.context.get('source', 'unknown') for m in metrics)),
                'last_update': max(m.timestamp for m in metrics) if metrics else 0,
                'status': self._get_overall_status(metrics)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Get dashboard summary error: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up analytics dashboard resources"""
        try:
            self.metrics_history.clear()
            self.dashboard_configs.clear()
            self.reports.clear()
            self.real_time_data.clear()
            self.alerts.clear()
            logger.info("Advanced Analytics Dashboard cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_analytics_dashboard() -> AdvancedAnalyticsDashboard:
    """Create and return analytics dashboard instance"""
    return AdvancedAnalyticsDashboard()

# Test function
if __name__ == "__main__":
    # Test the analytics dashboard
    dashboard = create_analytics_dashboard()
    
    # Add test metric data
    test_metric = MetricData(
        metric_id="test_metric_1",
        timestamp=time.time(),
        metric_type=MetricType.ENGAGEMENT,
        value=0.8,
        unit="score",
        context={"source": "visual"}
    )
    
    dashboard.add_metric_data("test_user", test_metric)
    
    # Create dashboard config
    config = dashboard.create_dashboard_config("test_user", "default")
    print(f"Dashboard config created: {config.get('config_id', 'unknown')}")
    
    # Generate report
    report = dashboard.generate_analytics_report("test_user", "test_session")
    print(f"Report generated: {report.report_id}")
    print(f"Summary keys: {list(report.summary.keys())}")
    
    # Get real-time dashboard
    real_time = dashboard.get_real_time_dashboard("test_user")
    print(f"Real-time dashboard status: {real_time.get('status', 'unknown')}")
    
    # Get summary
    summary = dashboard.get_dashboard_summary("test_user")
    print(f"Summary metrics: {summary.get('total_metrics', 0)}")
    
    # Cleanup
    dashboard.cleanup()
