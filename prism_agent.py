"""
prism Bridge Agent - ARC-AGI-3 Integration

Bridges the gap between prism's static grid prediction and ARC-AGI-3's interactive action-based environment.
Strategy: "Oracle + Hand" - prism predicts the target grid (Oracle), Python executes pixel-by-pixel (Hand).
"""

import json
import os
import subprocess
import tempfile
from typing import Optional, List, Tuple

try:
    from agents.agent import Agent
except ImportError:
    # Fallback for local testing without ARC-AGI-3 framework
    class Agent:
        MAX_ACTIONS = 80
        
class prismBridgeAgent(Agent):
    """
    ARC-AGI-3 Agent that uses prism as an Oracle to predict target grids,
    then compiles grid differences into GameAction sequences.
    
    Architecture:
      - prism (Rust): Static solver -> predicts target grid from training examples
      - Bridge Agent: Caches target, scans differences, emits PAINT actions
    """
    
    def __init__(self, *args, prism_bin: str = "./prism", **kwargs):
        super().__init__(*args, **kwargs) if hasattr(Agent, '__init__') else None
        
        self.prism_bin = prism_bin
        self.current_task_id: Optional[str] = None
        self.target_grid: Optional[List[List[int]]] = None
        self.training_examples: List[Tuple[List[List[int]], List[List[int]]]] = []
        
    def extract_grid_from_frame(self, frame) -> List[List[int]]:
        """Extract the main grid from FrameData (takes first layer)."""
        if hasattr(frame, 'frame') and frame.frame and len(frame.frame) > 0:
            return [row for row in frame.frame[0]]
        elif isinstance(frame, dict) and 'frame' in frame:
            return [row for row in frame['frame'][0]] if frame['frame'] else []
        return []
    
    def solve_task_with_prism(self, input_grid: List[List[int]], 
                              train_examples: Optional[List[Tuple]] = None) -> List[List[int]]:
        """Call prism engine to predict target grid from training examples."""
        
        if train_examples is None:
            train_examples = self.training_examples
            
        # Build prism JSON format
        task_data = {
            "task": {
                "train": [
                    {"input": inp, "output": out} 
                    for inp, out in train_examples
                ],
                "test": [{"input": input_grid, "output": None}]
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            inp_path = f"{tmpdir}/input.json"
            out_path = f"{tmpdir}/output.json"
            
            # Write input JSON
            with open(inp_path, 'w') as f:
                json.dump(task_data, f)
            
            # Execute prism engine
            try:
                result = subprocess.run(
                    [self.prism_bin, inp_path, out_path],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout per task
                )
            except subprocess.TimeoutExpired:
                print(f"⚠️ prism timed out for task {self.current_task_id}")
                return self._create_fallback_grid(input_grid)
            except FileNotFoundError:
                print(f"⚠️ prism binary not found at {self.prism_bin}")
                return self._create_fallback_grid(input_grid)
            
            # Read output JSON
            if not os.path.exists(out_path):
                return self._create_fallback_grid(input_grid)
                
            with open(out_path, 'r') as f:
                submission = json.load(f)
            
            # Parse prism output -> target grid
            # prism outputs in format: {"task_test_0": [[...]], "other_task_test_1": [...]}
            # Each value is the grid itself (list of rows), NOT a list of grids
            
            first_grid = None
            
            for key, grid in submission.items():
                if isinstance(grid, list) and len(grid) > 0:
                    # grid is the grid itself (list of rows)
                    if isinstance(grid[0], list):
                        first_grid = grid
                    break
            
            if not first_grid:
                print(f"⚠️ Failed to parse prism output, using fallback")
            
            return first_grid or self._create_fallback_grid(input_grid)
    
    def _create_fallback_grid(self, input_grid: List[List[int]]) -> List[List[int]]:
        """Create a fallback grid (zero-filled or copy of input)."""
        if not input_grid:
            return [[0 for _ in range(12)] for _ in range(12)]
        height = len(input_grid)
        width = len(input_grid[0]) if input_grid else 12
        return [[input_grid[y][x] for x in range(width)] for y in range(height)]
    
    def choose_action(self, frames: list, latest_frame) -> 'GameAction':
        """Core bridge logic - cache oracle and compile actions."""
        
        # Import GameAction dynamically (may not be available in local testing)
        try:
            from arcengine import GameAction, GameState
        except ImportError:
            # Fallback for local testing
            class MockGameAction:
                RESET = None
                PAINT = None
                
                def __init__(self, action_type: str):
                    self.type = action_type
                    self.data = {}
                    
                def set_data(self, data: dict):
                    self.data = data
                    
            class MockGameState:
                NOT_PLAYED = "NOT_PLAYED"
                GAME_OVER = "GAME_OVER"
                
            GameAction = MockGameAction  # type: ignore
            GameState = MockGameState
        
        current_grid = self.extract_grid_from_frame(latest_frame)
        
        # Handle game reset or game over state
        frame_state = getattr(latest_frame, 'state', None) if hasattr(latest_frame, 'state') else None
        
        # Check for NOT_PLAYED or GAME_OVER (handle both enum and string)
        if frame_state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:  # type: ignore
            self.target_grid = None
            return GameAction.RESET
        
        current_task_id = getattr(latest_frame, 'game_id', None) or (latest_frame.get('game_id') if isinstance(latest_frame, dict) else None)
        
        # Oracle call - only on first frame of each task
        if self.target_grid is None or current_task_id != self.current_task_id:
            print(f"🧠 [prism Oracle] Computing final answer for task {current_task_id}...")
            self.current_task_id = current_task_id
            
            # Solve using prism (pass training examples if available)
            self.target_grid = self.solve_task_with_prism(
                current_grid, 
                self.training_examples if hasattr(self, 'training_examples') else []
            )
            
            print(f"✅ [prism Oracle] Answer cached! Starting action sequence...")

        # Grid difference compiler - scan and emit PAINT actions
        if self.target_grid is None:
            return GameAction.RESET  # Safety fallback
        
        for y in range(len(current_grid)):
            if not current_grid:  # Empty grid check
                break
                
            for x in range(len(current_grid[0])):
                # Defensive bounds checking
                if y >= len(self.target_grid) or x >= len(self.target_grid[0]):
                    continue
                    
                if current_grid[y][x] != self.target_grid[y][x]:
                    # Found difference! Compile to PAINT action
                    try:
                        from arcengine import GameAction
                        
                        action = GameAction.PAINT  # type: ignore
                        if hasattr(action, 'set_data'):
                            action.set_data({
                                "x": x,
                                "y": y,
                                "color": self.target_grid[y][x],  # type: ignore
                                "reasoning": {"msg": f"prism Diff: patching ({x},{y})"}
                            })
                        else:
                            # Fallback for mock GameAction
                            action.data = {
                                "x": x,
                                "y": y,
                                "color": self.target_grid[y][x]  # type: ignore
                            }
                        return action
                    except ImportError:
                        # Mock implementation for local testing - use a simple dict as action placeholder
                        return {
                            "type": "PAINT",
                            "x": x,
                            "y": y,
                            "color": self.target_grid[y][x]  # type: ignore
                        }
        
        # No differences found - grid matches target but not WIN yet
        # This might mean we need a special "Submit" action or the game has additional steps
        
        try:
            from arcengine import GameAction, GameState
            
            # Check if game is actually won now
            current_state = getattr(latest_frame, 'state', None) if hasattr(latest_frame, 'state') else None
            levels_completed = getattr(latest_frame, 'levels_completed', 0) if hasattr(latest_frame, 'levels_completed') else None
            
            # If we've completed all levels, signal done
            if hasattr(latest_frame, 'win_levels') and levels_completed == getattr(latest_frame, 'win_levels', 0):
                print(f"🎯 All levels completed! Submitting scorecard.")
            else:
                # Try PAINT at (0, 0) with color -1 as a "done" signal
                action = GameAction.PAINT  # type: ignore
                if hasattr(action, 'set_data'):
                    action.set_data({
                        "x": 0, 
                        "y": 0, 
                        "color": -1,
                        "reasoning": {"msg": "No differences found, signaling done"}
                    })
                return action
                
        except ImportError:
            pass
        
        # Final fallback - reset to try again next frame
        return GameAction.RESET  # type: ignore
    
    def is_done(self, frames: list[object], latest_frame) -> bool:
        """Check if the game is complete."""
        frame_state = getattr(latest_frame, 'state', None) if hasattr(latest_frame, 'state') else None
        
        try:
            from arcengine import GameState
            
            # Check if game is won or completed all levels
            return frame_state == GameState.WIN  # type: ignore
            
        except ImportError:
            # Fallback for local testing - check if we've taken enough actions
            return len(frames) >= self.MAX_ACTIONS  # type: ignore


# ============================================================================
# Standalone Usage Example (without ARC-AGI-3 framework)
# ============================================================================

def standalone_example():
    """Example of using prismBridgeAgent outside the ARC-AGI-3 framework."""
    
    import os
    
    # Create a mock FrameData for testing
    class MockFrame:
        def __init__(self, game_id="test", grid=None):
            self.game_id = game_id
            if grid:
                self.frame = [grid]  # Wrap in list to match FrameData structure
    
    class MockGameState:
        NOT_PLAYED = "NOT_PLAYED"
        PLAYING = "PLAYING" 
        WIN = "WIN"
    
    # Initialize agent (without ARC-AGI-3 framework)
    import os
    macos_binary = "target/release/prism"
    agent = prismBridgeAgent(prism_bin=macos_binary) if os.path.exists(macos_binary) else None
    
    # Example: Process a single task
    if agent and os.path.exists("./prism"):
        # Setup training examples (these would normally come from ARC-AGI-3 environment)
        agent.training_examples = [
            (  # Example training pair
                [[0, 1, 0], [1, 2, 1], [0, 1, 0]],
                [[0, 2, 0], [2, 3, 2], [0, 2, 0]]
            )
        ]
        
        # Create input frame (current game state)
        current_grid = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
        frame = MockFrame(game_id="ar25", grid=current_grid)
        
        # Call choose_action (this will trigger prism Oracle on first call)
        action = agent.choose_action([frame], frame)
        
        print(f"Action type: {type(action)}")
        if isinstance(action, dict):
            print(f"Action data: {action}")
        elif hasattr(action, 'data'):
            print(f"Action data: {action.data}")
        else:
            print(f"Action: {action}")

if __name__ == "__main__":
    standalone_example()
