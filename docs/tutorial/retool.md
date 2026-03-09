### Tutorial: Customizing Tool Environment for ReTool on Qwen3-4B

This tutorial demonstrates how to create and customize a tool environment for implementing [ReTool: Reinforcement 
Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) - an approach that enhances long-form reasoning with tool-integrated learning. ReTool dynamically interleaves real-time code execution within natural language reasoning processes, making it an excellent showcase for customized tool environments.

The tutorial covers four main components:
1. Setting up the Sandbox environment for code execution
2. Creating custom tool environment for ReTool
3. Preparing datasets
4. Training your model using reinforcement learning

#### 1. Setting up the Sandbox Environment

##### Install SandboxFusion
We'll use SandboxFusion to provide a secure environment for code execution during training:

```bash
# Clone the SandboxFusion repository
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion

# Create a conda environment named "sandbox-runtime"
conda create -n sandbox-runtime python==3.11
conda activate sandbox-runtime

# Install dependencies
pip install -r runtime/python/requirements.txt
pip install poetry
poetry install

# Prepare and run the sandbox
mkdir -p docs/build
make run-online
```

The sandbox service should now be running and ready to handle code execution requests.

#### 2. Creating Custom Tool Environment for ReTool

ReTool implements a unique state transition function to integrate code execution with language model reasoning: During inference, the language model can request code execution through XML tags `<code></code>`. The code between these tags is extracted and executed in the sandbox environment, and the execution results are returned through `<interpreter></interpreter>`. After receiving the results, the language model continues reasoning by incorporating them, deciding whether to execute more code or provide a final answer.

##### Tool Environment Structure

The base tool environment is defined in `agent_r1/tool/base.py` and provides the interface for all custom tool environments:

```python
class BaseToolEnv(ABC):
    @abstractmethod
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """
        The State Transition Function of the Environment

        Args:
            raw_response: The raw response from the LLM
            
        Returns:
            tool_response: The tool response from the environment
            success: If the tool call is successful
            active: If the trajectory is actives
        """
        pass
    
    # Other important methods include batch_step, stop, extract_tool_calls, and format_tool_response
```

##### Understanding Tool Environments as State Transition Functions

To fully understand how to create custom tool environments, it's crucial to recognize their role in implementing the state transition function within the Markov Decision Process (MDP) framework for agent-based LLMs.

In traditional LLMs, the state transition is deterministic and straightforward:
1. The model generates a token
2. The token is added to the existing sequence
3. This augmented sequence becomes the new state

However, for agent-based LLMs that can interact with external tools, this process becomes more complex and stochastic:
1. The model generates tokens that may include tool-triggering patterns (like `<code>...</code>` in ReTool)
2. When such patterns are detected, the environment extracts the tool call
3. The tool executes the request in the external environment (introducing non-determinism)
4. The tool's response is formatted and returned to the model
5. The model continues generation based on this new information

The tool environment is precisely an abstraction of this non-deterministic state transition process. It defines:
- How to recognize when a tool should be called (`process_responses_ids`) - or equivalently, when to terminate a single round of generation to execute a tool
- How to extract parameters from the model's response for tool execution (`extract_tool_calls`)
- How to format the results for the model to consume (`format_tool_response`)
- When to stop the generation process (`stop`)

These abstract methods are then combined in the core `step` method, which implements the complete state transition function. The `step` method:
1. Takes the raw LLM response as input
2. Extracts tool parameters using `extract_tool_calls`
3. Passes these parameters to the appropriate tool for execution
4. Formats the execution results using `format_tool_response`
5. Returns the formatted response along with success and activity status flags

This unified approach allows for a clean separation of concerns while maintaining a cohesive state transition process that can be customized for different tool interaction patterns.

Thus, when creating a custom tool environment, you are essentially implementing a specific type of state transition function that dictates how your agent interacts with external tools and incorporates their responses into its reasoning process.

##### Implementing ReTool Environment

Now that we understand the theoretical basis of tool environments as state transition functions, let's examine how ReTool concretely implements these concepts. The ReTool environment is a practical example of translating the MDP framework into code.

```python
class ReToolEnv(BaseToolEnv):
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int):
        self.tool = tools[0]
        assert self.tool.name == "python"
        self.max_tool_response_length = max_tool_response_length
```

In the initialization, ReTool accepts tools (specifically a Python executor) and sets limits for response length, establishing the environment's parameters.

Let's see how ReTool implements each component of the non-deterministic state transition process:

> **Special Note on `process_responses_ids`**: In most cases, tool calls are triggered when the model generates specific tokens or special strings. For these common scenarios, we can simply configure generation parameters like `stop_token_ids` or `stop` (e.g., in vLLM) to terminate generation at the appropriate point, without implementing a custom `process_responses_ids` method. In ReTool, we can simply set `stop=["</code>"]` to terminate generation when the code block ends. The `process_responses_ids` method is only needed for more complex cases where sophisticated pattern recognition is required to identify when to trigger tool execution.

1. **Extracting Parameters for Tool Execution** (`extract_tool_calls`):

```python
def extract_tool_calls(self, raw_response: str) -> List[Any]:
    """
    extract the code after "<code>", and before "</code>"
    """
    code = ''
    start = False
    for line in raw_response.split('\n'):
        if line.startswith('<code>'):
            code += '\n# ========\n'
            start = True
        elif line.startswith('</code>'):
            start = False
        elif start:
            if line.startswith('```'):
                continue
            code += line + '\n'
    if start or len(code) == 0:
        # the code is incomplete
        return []
    return [code]
```

This method identifies and extracts the code surrounded by `<code>...</code>` tags in the LLM's response. It doesn't execute the code itself but prepares the parameters (code to be executed) for the actual tool execution that happens later.

2. **Formatting Results for the Model** (`format_tool_response`):

```python
def format_tool_response(self, tool_responses: List[str]) -> str:
    if len(tool_responses) == 0:
        return ""
    if len(tool_responses[0]) > self.max_tool_response_length:
        tool_responses[0] = tool_responses[0][:self.max_tool_response_length] + "..."
    return "\n<interpreter>\n" + tool_responses[0] + "\n</interpreter>\n"
```

This method formats the execution results into a structure that the LLM can recognize and incorporate into its reasoning, wrapping the output in `<interpreter></interpreter>` tags.

3. **Determining When to Stop Generation** (`stop`):

```python
def stop(self, raw_response: str) -> bool:
    tool_calls = self.extract_tool_calls(raw_response)
    if len(tool_calls) == 0:
        return True
    else:
        return False
```

The `stop` method decides when to end generation - in this case, when there are no more code blocks to execute, indicating that the model has completed its reasoning or provided a final answer.

4. **Orchestrating the Complete State Transition** (the `step` method):

```python
def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
    code = self.extract_tool_calls(raw_response)
    if len(code) == 0:
        return "", [], False
    code = code[0]
    tool_response, tool_success = self.tool.execute({"code": code})
    tool_response = self.format_tool_response([tool_response])
    return tool_response, [tool_success], True
```

The `step` method coordinates the entire state transition process:
- First, it extracts the code parameters using `extract_tool_calls`
- If valid code is found, it passes these parameters to the actual tool for execution
- The execution happens in the tool itself (`self.tool.execute`), not in the environment
- It then formats the results and returns them with status information

The ReTool environment also implements a `batch_step` method for efficient batch processing during training. Through these components, ReTool creates a complete implementation of the non-deterministic state transition function needed for code-executing agents.

This implementation enables the "think-execute-think" cycle at the core of ReTool: the LLM can reason about a problem, write code to solve parts of it, observe the execution results, and continue reasoning based on this new information.

##### Registering the Tool Environment

Tool environments are registered in `agent_r1/tool/envs/__init__.py` to make them available for use:

```python
from agent_r1.tool.envs.retool import ReToolEnv

def _default_env(name):
    # ...
    elif name == "retool":
        return ReToolEnv
    # ...
```

This factory pattern allows your code to create the appropriate tool environment based on configuration.

#### 3. Preparing Data

```bash
# Create data directory
mkdir -p data/retool

# Run preprocessing script
python examples/data_preprocess/retool.py --local_dir ./data/retool
```

The preprocessing script will automatically download and prepare the ReTool dataset for training.

#### 4. Training the Model

ReTool uses a special format to trigger code execution tools, as described in the original paper. We'll first use a small amount of data for a cold start. A fine-tuned model is available for download at `russwest404/Qwen3-4B-ReTool-SFT`.

##### Prepare Training Script
Copy the training script to the main directory:

```bash
cp examples/trainer/run_ppo_retool.sh ./
```

##### Important Configuration Update
Before running the training script, you need to modify the configuration file to properly handle the `</code>` stop character. In bash scripts, this character can be incorrectly parsed.

```bash
# Edit the agent trainer configuration file
nano agent_r1/src/config/agent_trainer.yaml
```

In the configuration file, find the `actor_rollout_ref.rollout` section and update the `stop` parameter. You should add `"</code>"` to the stop list:

```yaml
  rollout:
    # ... existing configuration ...
    stop: ["</code>"]
```

Save the file after making this change.

> **Note**: A more elegant solution would be to directly configure this parameter in the training script itself, avoiding the need for manual configuration. If you discover a method to set `actor_rollout_ref.rollout.stop` directly in the script (for example, by using environment variables or command-line arguments), we welcome your pull request contributions to improve this workflow.

##### Configure Training Parameters
Edit the `run_ppo_retool.sh` script to adjust parameters according to your needs, ensuring that the tool environment is set to "retool".

##### Run Training
Execute the training script:

```bash
bash run_ppo_retool.sh
```

Training progress and logs will be saved to Wandb (if configured) and the console.
