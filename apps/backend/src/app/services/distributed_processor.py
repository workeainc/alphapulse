import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import aiohttp
import psutil

logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ProcessingNode:
    node_id: str
    host: str
    port: int
    status: NodeStatus
    capacity: int  # Max concurrent tasks
    current_load: int
    last_heartbeat: datetime
    performance_score: float
    metadata: Dict[str, Any]

@dataclass
class DistributedTask:
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    assigned_node: Optional[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]

class DistributedProcessor:
    """
    Coordinates distributed pattern processing across multiple servers
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, ProcessingNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue: List[str] = []
        self.load_balancer = LoadBalancer()
        self.failover_manager = FailoverManager()
        self.monitoring = DistributedMonitoring()
        
        # Performance tracking
        self.stats = {
            'total_tasks_processed': 0,
            'total_nodes_managed': 0,
            'average_processing_time': 0.0,
            'system_throughput': 0.0,
            'failover_count': 0,
            'load_balancing_efficiency': 0.0
        }
        
        # Configuration
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        self.task_timeout = config.get('task_timeout', 300)
        self.max_retries = config.get('max_retries', 3)
        self.enable_auto_scaling = config.get('enable_auto_scaling', True)
        
        logger.info("DistributedProcessor initialized with config: %s", config)
    
    async def start(self):
        """Start the distributed processor"""
        logger.info("Starting distributed processor...")
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._task_monitor())
        asyncio.create_task(self._performance_monitor())
        
        if self.enable_auto_scaling:
            asyncio.create_task(self._auto_scaling_monitor())
        
        logger.info("Distributed processor started successfully")
    
    def register_node(self, node_info: Dict[str, Any]) -> str:
        """Register a new processing node"""
        node_id = str(uuid.uuid4())
        
        node = ProcessingNode(
            node_id=node_id,
            host=node_info['host'],
            port=node_info['port'],
            status=NodeStatus.ONLINE,
            capacity=node_info.get('capacity', 10),
            current_load=0,
            last_heartbeat=datetime.now(timezone.utc),
            performance_score=1.0,
            metadata=node_info.get('metadata', {})
        )
        
        self.nodes[node_id] = node
        self.stats['total_nodes_managed'] = len(self.nodes)
        
        logger.info("Registered new node: %s (%s:%s)", node_id, node.host, node.port)
        return node_id
    
    def submit_task(self, task_type: str, data: Dict[str, Any], 
                   priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit a new task for distributed processing"""
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            assigned_node=None,
            status='pending',
            created_at=datetime.now(timezone.utc),
            started_at=None,
            completed_at=None,
            result=None,
            error=None
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda tid: self.tasks[tid].priority.value, reverse=True)
        
        logger.info("Submitted task %s (type: %s, priority: %s)", 
                   task_id, task_type, priority.name)
        
        return task_id
    
    async def _heartbeat_monitor(self):
        """Monitor node heartbeats and update status"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                for node_id, node in self.nodes.items():
                    # Check if node is responsive
                    if (current_time - node.last_heartbeat).seconds > self.heartbeat_interval * 2:
                        if node.status != NodeStatus.OFFLINE:
                            await self._mark_node_offline(node_id)
                    
                    # Update performance score based on current load
                    load_ratio = node.current_load / node.capacity if node.capacity > 0 else 1.0
                    node.performance_score = max(0.1, 1.0 - load_ratio)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error("Error in heartbeat monitor: %s", e)
                await asyncio.sleep(5)
    
    async def _task_monitor(self):
        """Monitor task execution and handle timeouts"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Process pending tasks
                await self._process_pending_tasks()
                
                # Check for timed out tasks
                for task_id, task in self.tasks.items():
                    if (task.status == 'running' and task.started_at and
                        (current_time - task.started_at).seconds > self.task_timeout):
                        await self._handle_task_timeout(task_id)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error("Error in task monitor: %s", e)
                await asyncio.sleep(5)
    
    async def _process_pending_tasks(self):
        """Process pending tasks by assigning them to available nodes"""
        if not self.task_queue:
            return
        
        available_nodes = [
            node_id for node_id, node in self.nodes.items()
            if (node.status == NodeStatus.ONLINE and 
                node.current_load < node.capacity)
        ]
        
        if not available_nodes:
            return
        
        # Get next task from queue
        task_id = self.task_queue.pop(0)
        task = self.tasks[task_id]
        
        # Select best node using load balancer
        selected_node = self.load_balancer.select_node(available_nodes, self.nodes)
        
        if selected_node:
            await self._assign_task_to_node(task_id, selected_node)
    
    async def _assign_task_to_node(self, task_id: str, node_id: str):
        """Assign a task to a specific node"""
        task = self.tasks[task_id]
        node = self.nodes[node_id]
        
        task.assigned_node = node_id
        task.status = 'running'
        task.started_at = datetime.now(timezone.utc)
        
        node.current_load += 1
        node.status = NodeStatus.BUSY if node.current_load >= node.capacity else NodeStatus.ONLINE
        
        logger.info("Assigned task %s to node %s", task_id, node_id)
        
        # Send task to node (in real implementation, this would be HTTP/WebSocket)
        await self._send_task_to_node(node_id, task)
    
    async def _send_task_to_node(self, node_id: str, task: DistributedTask):
        """Send task data to processing node"""
        # This is a placeholder - in real implementation, you'd send via HTTP/WebSocket
        logger.debug("Sending task %s to node %s", task.task_id, node_id)
    
    async def _handle_task_timeout(self, task_id: str):
        """Handle task timeout by reassigning or marking as failed"""
        task = self.tasks[task_id]
        node_id = task.assigned_node
        
        if node_id and node_id in self.nodes:
            # Reduce load on the node
            self.nodes[node_id].current_load = max(0, self.nodes[node_id].current_load - 1)
        
        # Mark task as failed
        task.status = 'failed'
        task.error = 'Task timeout'
        task.completed_at = datetime.now(timezone.utc)
        
        logger.warning("Task %s timed out", task_id)
    
    async def _mark_node_offline(self, node_id: str):
        """Mark a node as offline and handle failover"""
        node = self.nodes[node_id]
        node.status = NodeStatus.OFFLINE
        
        # Handle failover for tasks assigned to this node
        await self.failover_manager.handle_node_failure(node_id, self.tasks, self.nodes)
        
        logger.warning("Node %s marked as offline", node_id)
    
    async def _performance_monitor(self):
        """Monitor overall system performance"""
        while True:
            try:
                # Calculate performance metrics
                total_load = sum(node.current_load for node in self.nodes.values())
                total_capacity = sum(node.capacity for node in self.nodes.values())
                
                if total_capacity > 0:
                    self.stats['load_balancing_efficiency'] = 1.0 - (total_load / total_capacity)
                
                # Update monitoring
                await self.monitoring.update_metrics(self.stats, self.nodes, self.tasks)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error("Error in performance monitor: %s", e)
                await asyncio.sleep(5)
    
    async def _auto_scaling_monitor(self):
        """Monitor system load and trigger auto-scaling if needed"""
        while True:
            try:
                # Check if we need more capacity
                total_load = sum(node.current_load for node in self.nodes.values())
                total_capacity = sum(node.capacity for node in self.nodes.values())
                
                if total_capacity > 0 and (total_load / total_capacity) > 0.8:
                    await self._trigger_auto_scaling()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error("Error in auto-scaling monitor: %s", e)
                await asyncio.sleep(10)
    
    async def _trigger_auto_scaling(self):
        """Trigger auto-scaling to add more processing capacity"""
        logger.info("Triggering auto-scaling due to high load")
        # In real implementation, this would create new nodes or containers
        
        self.stats['auto_scaling_triggered'] = self.stats.get('auto_scaling_triggered', 0) + 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'total_nodes': len(self.nodes),
            'online_nodes': len([n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]),
            'total_tasks': len(self.tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == 'pending']),
            'running_tasks': len([t for t in self.tasks.values() if t.status == 'running']),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == 'completed']),
            'failed_tasks': len([t for t in self.tasks.values() if t.status == 'failed']),
            'stats': self.stats,
            'monitoring': self.monitoring.get_status()
        }
    
    async def stop(self):
        """Stop the distributed processor"""
        logger.info("Stopping distributed processor...")
        # Cleanup tasks would go here
        logger.info("Distributed processor stopped")

class LoadBalancer:
    """Intelligent load balancing for distributed tasks"""
    
    def select_node(self, available_nodes: List[str], 
                   all_nodes: Dict[str, ProcessingNode]) -> Optional[str]:
        """Select the best node for a task using weighted round-robin"""
        if not available_nodes:
            return None
        
        # Calculate weighted scores based on performance and load
        node_scores = []
        for node_id in available_nodes:
            node = all_nodes[node_id]
            load_ratio = node.current_load / node.capacity if node.capacity > 0 else 1.0
            score = node.performance_score * (1.0 - load_ratio)
            node_scores.append((node_id, score))
        
        # Sort by score (highest first)
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best node
        return node_scores[0][0] if node_scores else None

class FailoverManager:
    """Handles failover when nodes go offline"""
    
    async def handle_node_failure(self, failed_node_id: str, 
                                tasks: Dict[str, DistributedTask],
                                nodes: Dict[str, ProcessingNode]):
        """Handle failover for tasks assigned to a failed node"""
        failed_tasks = [
            task_id for task_id, task in tasks.items()
            if task.assigned_node == failed_node_id and task.status == 'running'
        ]
        
        for task_id in failed_tasks:
            task = tasks[task_id]
            task.status = 'pending'
            task.assigned_node = None
            task.started_at = None
            
            # Re-queue the task
            # In real implementation, this would be more sophisticated
        
        logger.info("Handled failover for %d tasks from failed node %s", 
                   len(failed_tasks), failed_node_id)

class DistributedMonitoring:
    """Monitors distributed system performance"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.last_update = datetime.now(timezone.utc)
    
    async def update_metrics(self, stats: Dict[str, Any], 
                           nodes: Dict[str, ProcessingNode],
                           tasks: Dict[str, DistributedTask]):
        """Update monitoring metrics"""
        current_time = datetime.now(timezone.utc)
        
        metrics = {
            'timestamp': current_time,
            'node_count': len(nodes),
            'task_count': len(tasks),
            'system_load': sum(node.current_load for node in nodes.values()),
            'system_capacity': sum(node.capacity for node in nodes.values()),
            'performance_score': sum(node.performance_score for node in nodes.values()) / len(nodes) if nodes else 0,
            'stats': stats.copy()
        }
        
        self.metrics_history.append(metrics)
        self.last_update = current_time
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'last_update': self.last_update.isoformat(),
            'metrics_count': len(self.metrics_history),
            'alerts_count': len(self.alerts),
            'recent_metrics': self.metrics_history[-10:] if self.metrics_history else []
        }
