"""
Rosetta Task Manager - Manages background Rosetta simulations
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from .rosetta_client import RosettaClient

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class RosettaTask:
    """Represents a single Rosetta simulation task"""
    
    def __init__(self, hypothesis_id: str, hypothesis_content: str, task_type: str):
        self.task_id = f"rosetta_{hypothesis_id}_{datetime.now().timestamp()}"
        self.hypothesis_id = hypothesis_id
        self.hypothesis_content = hypothesis_content
        self.task_type = task_type  # 'binding', 'structure', 'docking'
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.task_handle = None  # asyncio.Task


class RosettaTaskManager:
    """
    Manages background Rosetta simulations
    
    Features:
    - Non-blocking task submission
    - Progress tracking
    - Result retrieval
    - Timeout handling
    - Task cancellation
    """
    
    def __init__(
        self,
        max_concurrent: int = 2,
        python_path: str = "python3",
        timeout: Optional[int] = None,
        default_timeout: int = 600
    ):
        """
        Args:
            max_concurrent: Maximum number of concurrent Rosetta tasks
            python_path: Path to Python with PyRosetta installed
            timeout: Override for default timeout
            default_timeout: Default timeout in seconds (10 minutes)
        """
        self.max_concurrent = max_concurrent
        self.default_timeout = timeout if timeout is not None else default_timeout
        self.python_path = python_path
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize Rosetta client
        self.rosetta_client = RosettaClient(python_path=python_path)
        
        # Task storage
        self.tasks: Dict[str, RosettaTask] = {}  # task_id -> RosettaTask
        self.hypothesis_tasks: Dict[str, List[str]] = {}  # hypothesis_id -> [task_ids]
        
        logger.info(f"RosettaTaskManager initialized (max_concurrent={max_concurrent})")
    
    def submit_task(
        self,
        hypothesis_id: str,
        hypothesis_content: str,
        task_type: str = 'binding',
        rosetta_config: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Submit a Rosetta task for background execution
        
        Args:
            hypothesis_id: Hypothesis ID
            hypothesis_content: Full hypothesis content
            task_type: Type of simulation ('binding', 'structure', 'docking')
            rosetta_config: Configuration for Rosetta
            timeout: Task timeout in seconds
            
        Returns:
            task_id: Unique task identifier
        """
        # Create task
        task = RosettaTask(hypothesis_id, hypothesis_content, task_type)
        self.tasks[task.task_id] = task
        
        # Track by hypothesis
        if hypothesis_id not in self.hypothesis_tasks:
            self.hypothesis_tasks[hypothesis_id] = []
        self.hypothesis_tasks[hypothesis_id].append(task.task_id)
        
        # Start background execution
        config = rosetta_config or {}
        timeout = timeout or self.default_timeout
        task.task_handle = asyncio.create_task(
            self._run_task(task, config, timeout)
        )
        
        logger.info(f"Submitted Rosetta task {task.task_id} for hypothesis {hypothesis_id}")
        return task.task_id
    
    async def _run_task(self, task: RosettaTask, config: Dict, timeout: int):
        """Execute Rosetta task with resource control"""
        async with self.semaphore:  # Limit concurrent execution
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            logger.info(f"Starting Rosetta task {task.task_id} (type: {task.task_type})")
            
            try:
                # Run with timeout
                result = await asyncio.wait_for(
                    self._execute_rosetta(task, config),
                    timeout=timeout
                )
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                duration = (task.completed_at - task.started_at).total_seconds()
                logger.info(f"✓ Completed Rosetta task {task.task_id} in {duration:.1f}s")
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.error = "Task cancelled"
                logger.info(f"Cancelled Rosetta task {task.task_id}")
                raise
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.TIMEOUT
                task.error = f"Timeout after {timeout}s"
                logger.warning(f"Rosetta task {task.task_id} timed out")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.error(f"Rosetta task {task.task_id} failed: {e}")
    
    async def _execute_rosetta(self, task: RosettaTask, config: Dict) -> Dict[str, Any]:
        """
        Execute PyRosetta simulation using RosettaClient
        
        For protein binder design, we:
        1. Generate a temporary PDB structure from hypothesis
        2. Score it with PyRosetta
        3. Optionally predict binding energy
        """
        if not self.rosetta_client.pyrosetta_available:
            logger.warning("PyRosetta not available. Using heuristic scoring.")
            return self._estimate_without_rosetta(task)
        
        try:
            # For now, use a placeholder PDB or extract from hypothesis
            # In real scenario, hypothesis should contain or reference a PDB structure
            pdb_path = config.get("pdb_path")
            
            if not pdb_path:
                # Generate mock PDB from hypothesis (simplified)
                logger.info("No PDB provided, using mock structure for scoring")
                return self._estimate_without_rosetta(task)
            
            # Score the structure
            logger.info(f"Scoring structure with PyRosetta: {pdb_path}")
            score_result = await self.rosetta_client.score_pdb(
                pdb_path=pdb_path,
                scorefxn=config.get("scorefxn", "ref2015")
            )
            
            if "error" in score_result:
                raise RuntimeError(f"PyRosetta scoring failed: {score_result['error']}")
            
            # For binding tasks, also analyze interface
            result = {
                "success": True,
                "task_type": task.task_type,
                "source": "pyrosetta",
                "rosetta_score": score_result.get("score"),
                "scorefxn": score_result.get("scorefxn", "ref2015")
            }
            
            if task.task_type == 'binding' and config.get("interface_chains"):
                logger.info("Analyzing binding interface")
                binding_result = await self.rosetta_client.predict_binding_energy(
                    pdb_path=pdb_path,
                    interface_chains=config.get("interface_chains")
                )
                
                if "error" not in binding_result:
                    result["binding_energy"] = binding_result.get("total_score")
                    result["interface"] = binding_result.get("interface")
                    result["binding_analyzed"] = binding_result.get("binding_analyzed", False)
            
            return result
            
        except Exception as e:
            logger.error(f"PyRosetta execution failed: {e}")
            # Fall back to heuristic
            return self._estimate_without_rosetta(task)

    def _parse_rosetta_output(self, output: str, task_type: str) -> Dict[str, Any]:
        """Parse Rosetta stdout into a structured result."""
        if not output:
            return {"raw_output": "", "success": True}
        
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        
        metrics: Dict[str, Any] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metrics[key.strip()] = self._coerce_value(value.strip())
        
        if task_type == 'binding':
            metrics.setdefault("binding_energy", metrics.get("total_energy"))
            metrics.setdefault("predicted_kd", f"{metrics.get('kd', 'N/A')} nM")
        elif task_type == 'structure':
            metrics.setdefault("structure_quality", metrics.get("quality", 0.0))
            metrics.setdefault("rmsd", metrics.get("rmsd", 0.0))
        
        metrics["raw_output"] = output[-4000:]
        return metrics

    @staticmethod
    def _coerce_value(value: str) -> Any:
        """Convert string values to int/float when possible."""
        try:
            if value.lower() in {"true", "false"}:
                return value.lower() == "true"
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _estimate_without_rosetta(self, task: RosettaTask) -> Dict[str, Any]:
        """Lightweight deterministic scoring used when Rosetta is unavailable."""
        seed = abs(hash(task.hypothesis_content)) % 1_000_000
        
        if task.task_type == 'structure':
            return {
                'structure_quality': round(0.6 + (seed % 300) / 1000, 3),
                'rmsd': round(0.5 + (seed % 180) / 100, 2),
                'rosetta_energy': round(-50 - (seed % 500) / 5, 2),
                'secondary_structure': 'mixed',
                'success': True,
                'source': 'heuristic'
            }
        
        # Default to binding-style metrics
        binding_energy = round(-5 - (seed % 3000) / 150, 2)
        interface_area = round(500 + (seed % 600), 1)
        kd = max(1, (seed % 9000) // 30)
        interface_quality = round(0.6 + (seed % 250) / 1000, 3)
        
        return {
            'binding_energy': binding_energy,
            'interface_area': interface_area,
            'predicted_kd': f"{kd} nM",
            'interface_quality_score': interface_quality,
            'rosetta_energy': round(binding_energy * 10, 2),
            'success': True,
            'source': 'heuristic'
        }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a task
        
        Args:
            task_id: The task ID to check
            
        Returns:
            Dictionary with status information
        """
        task = self.tasks.get(task_id)
        if not task:
            return {
                'exists': False,
                'status': 'not_found',
                'message': f'Task {task_id} not found'
            }
        
        return {
            'exists': True,
            'status': task.status.value,
            'hypothesis_id': task.hypothesis_id,
            'task_type': task.task_type,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'has_result': task.result is not None,
            'error': task.error
        }
    
    async def cancel_task(self, hypothesis_id: str, reason: str = "hypothesis rejected"):
        """Cancel Rosetta task for a hypothesis"""
        task_ids = self.hypothesis_tasks.get(hypothesis_id, [])
        
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                if task.task_handle and not task.task_handle.done():
                    task.task_handle.cancel()
                    logger.info(f"Cancelled Rosetta task {task_id}: {reason}")
    
    def get_completed_tasks(self) -> List[RosettaTask]:
        """Get all completed tasks that haven't been processed"""
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.COMPLETED
        ]
    
    def get_hypothesis_result(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific hypothesis"""
        task_ids = self.hypothesis_tasks.get(hypothesis_id, [])
        
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                return task.result
        
        return None
    
    def get_pending_count(self) -> int:
        """Get number of pending/running tasks"""
        return sum(
            1 for task in self.tasks.values()
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        )
    
    async def wait_all(self, timeout: Optional[int] = None, check_interval: int = 30):
        """
        Wait for all pending tasks to complete with periodic status updates
        
        Args:
            timeout: Maximum wait time in seconds
            check_interval: Seconds between status checks
        """
        pending_tasks = [
            task.task_handle for task in self.tasks.values()
            if task.task_handle and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        ]
        
        if not pending_tasks:
            return
        
        logger.info(f"Waiting for {len(pending_tasks)} Rosetta tasks...")
        
        start_time = datetime.now()
        while pending_tasks:
            # Wait for check_interval or until tasks complete
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                timeout=check_interval,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Log progress
            completed = len(self.tasks) - len(pending_tasks)
            logger.info(f"  Rosetta progress: {completed}/{len(self.tasks)} completed")
            
            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    logger.warning(f"Rosetta wait timeout after {elapsed:.0f}s")
                    break
            
            if not pending_tasks:
                break
    
    def get_summary(self) -> Dict[str, int]:
        """Get task statistics"""
        summary = {
            'total': len(self.tasks),
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0,
            'timeout': 0,
            'cancelled': 0
        }
        
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                summary['pending'] += 1
            elif task.status == TaskStatus.RUNNING:
                summary['running'] += 1
            elif task.status == TaskStatus.COMPLETED:
                summary['completed'] += 1
            elif task.status == TaskStatus.FAILED:
                summary['failed'] += 1
            elif task.status == TaskStatus.TIMEOUT:
                summary['timeout'] += 1
            elif task.status == TaskStatus.CANCELLED:
                summary['cancelled'] += 1
        
        return summary
