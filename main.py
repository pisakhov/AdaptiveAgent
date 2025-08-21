from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from rich import print
from rich.console import Console
from rich.table import Table
from rich import box
import random
import time
import threading
import builtins
import inspect
from functools import wraps
from datetime import datetime

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

console = Console()

def get_score_color(score):
    """Get color based on boolean score: green=True, red=False"""
    return "green" if score else "red"

def print_performance_table(prompts, scores_response, scores_chain_of_thought, scores_prompt, messages_counts):
    """Print minimalist performance table with color coding"""
    pass_rates = []
    for i in range(len(scores_response)):
        passes = sum([scores_response[i], scores_chain_of_thought[i], scores_prompt[i]])
        pass_rate = (passes / 3) * 100
        pass_rates.append(pass_rate)
    best_avg_index = pass_rates.index(max(pass_rates)) if pass_rates else 0
    table = Table(box=box.ROUNDED)
    table.add_column("Attempt", style="cyan", width=8)
    table.add_column("Prompt", style="magenta", width=45)
    table.add_column("Messages", justify="center", style="blue", width=8)
    table.add_column("Response", justify="right", width=8)
    table.add_column("CoT", justify="right", width=8)
    table.add_column("Prompt", justify="right", width=8)
    table.add_column("Pass Rate", justify="right", style="bold", width=10)
    for i in range(len(scores_response)):
        pass_rate = pass_rates[i]
        prompt_text = prompts[i] if i < len(prompts) else "N/A"
        short_prompt = prompt_text[:40] + "..." if len(prompt_text) > 40 else prompt_text
        attempt_text = f"{i+1}/{len(scores_response)}"
        if i == best_avg_index:
            attempt_text += " 🏆"
        msg_count = messages_counts[i] if i < len(messages_counts) else 0
        response_color = get_score_color(scores_response[i])
        cot_color = get_score_color(scores_chain_of_thought[i])
        prompt_color = get_score_color(scores_prompt[i])
        pass_rate_color = "green" if pass_rate >= 100 else "yellow" if pass_rate >= 50 else "red"
        response_mark = "✓" if scores_response[i] else "✗"
        cot_mark = "✓" if scores_chain_of_thought[i] else "✗"
        prompt_mark = "✓" if scores_prompt[i] else "✗"
        table.add_row(
            attempt_text,
            short_prompt,
            str(msg_count),
            f"[{response_color}]{response_mark}[/{response_color}]",
            f"[{cot_color}]{cot_mark}[/{cot_color}]",
            f"[{prompt_color}]{prompt_mark}[/{prompt_color}]",
            f"[bold {pass_rate_color}]{pass_rate:.0f}%[/bold {pass_rate_color}]"
        )
    console.print("\n")
    console.print(table)

def vero_agent(message: str | None = None):
    """Visualize node execution with colored Rich output"""
    if not hasattr(vero_agent, "_lock"):
        vero_agent._lock = threading.Lock()
        vero_agent._colors = {}
        vero_agent._palette = [
            "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
            "bright_blue", "bright_red", "cyan", "green", "yellow", "magenta",
            "blue", "red", "bright_white", "orange1", "purple", "violet",
            "spring_green1", "dodger_blue1", "deep_pink1", "gold1", "orchid1"
        ]
    verbs = ["Analyzing", "Processing", "Computing", "Evaluating", "Optimizing", "Synthesizing", "Reasoning", "Learning"]
    def color_for(name: str) -> str:
        if name not in vero_agent._colors:
            idx = len(vero_agent._colors) % len(vero_agent._palette)
            vero_agent._colors[name] = vero_agent._palette[idx]
        return vero_agent._colors[name]
    def decorator(func):
        is_async = inspect.iscoroutinefunction(func)
        async def async_wrap(*args, **kwargs):
            name = func.__name__
            color = color_for(name)
            action = message or random.choice(verbs)
            title = " ".join(w.capitalize() for w in name.split("_"))
            ts = datetime.now().strftime("%H:%M:%S")
            with vero_agent._lock:
                console.print(f"[{color}]● {title}[/{color}] [dim]{ts}[/dim] — [italic]{action}[/italic]")
            start = time.time()
            orig_print = builtins.print
            def indented_print(*a, **k):
                text = " ".join(str(x) for x in a)
                for line in text.split("\n"):
                    with vero_agent._lock:
                        console.print(f"  [{color}]│[/{color}] [bright_black]{line}[/bright_black]")
            builtins.print = indented_print
            try:
                return await func(*args, **kwargs)
            finally:
                builtins.print = orig_print
                elapsed = time.time() - start
                ts2 = datetime.now().strftime("%H:%M:%S")
                with vero_agent._lock:
                    console.print(f"[{color}]✓ {title}[/{color}] [dim]{elapsed:.2f}s at {ts2}[/dim]\n")
        def sync_wrap(*args, **kwargs):
            name = func.__name__
            color = color_for(name)
            action = message or random.choice(verbs)
            title = " ".join(w.capitalize() for w in name.split("_"))
            ts = datetime.now().strftime("%H:%M:%S")
            with vero_agent._lock:
                console.print(f"[{color}]● {title}[/{color}] [dim]{ts}[/dim] — [italic]{action}[/italic]")
            start = time.time()
            orig_print = builtins.print
            def indented_print(*a, **k):
                text = " ".join(str(x) for x in a)
                for line in text.split("\n"):
                    with vero_agent._lock:
                        console.print(f"  [{color}]│[/{color}] [bright_black]{line}[/bright_black]")
            builtins.print = indented_print
            try:
                return func(*args, **kwargs)
            finally:
                builtins.print = orig_print
                elapsed = time.time() - start
                ts2 = datetime.now().strftime("%H:%M:%S")
                with vero_agent._lock:
                    console.print(f"[{color}]✓ {title}[/{color}] [dim]{elapsed:.2f}s at {ts2}[/dim]\n")
        return wraps(func)(async_wrap if is_async else sync_wrap)
    return decorator

@tool
def get_location_info(location: str) -> str:
    """Get comprehensive location information including time, timezone, date, weather

    Uses free APIs (no API key required) for real weather and location data.
    Supports any location format: cities, addresses, coordinates, airport codes.

    Args:
        location: Any location (e.g., "New York", "London", "Tokyo", "Paris, France",
                 "40.7128,-74.0060", "JFK", "Eiffel Tower")

    Returns:
        JSON string with time, timezone, date, day of week, temperature, and weather info
    """
    print(f"[dim]🌍 Tool Call: get_location_info({location})[/dim]")

    try:
        import requests
        import json
        from datetime import datetime
        import pytz

        # Step 1: Get detailed location info using Nominatim (free geocoding)
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        nominatim_params = {
            'q': location,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1
        }

        # Nominatim requires a User-Agent header
        headers = {
            'User-Agent': 'AdaptiveAgent/1.0 (https://github.com/adaptive-agent)'
        }

        try:
            geo_response = requests.get(nominatim_url, params=nominatim_params,
                                      headers=headers, timeout=10)

            if geo_response.status_code == 200:
                geo_data = geo_response.json()

                if geo_data:
                    geo_info = geo_data[0]
                    lat = float(geo_info['lat'])
                    lon = float(geo_info['lon'])
                    address = geo_info.get('address', {})

                    # Extract location details
                    city = (address.get('city') or address.get('town') or
                           address.get('village') or address.get('hamlet') or
                           address.get('municipality') or
                           geo_info.get('display_name', '').split(',')[0])
                    country = address.get('country', 'Unknown')
                    state = (address.get('state') or address.get('province') or
                            address.get('region') or address.get('county', ''))

                    location_details = {
                        "name": city,
                        "country": country,
                        "state": state,
                        "coordinates": {"lat": lat, "lon": lon},
                        "full_address": geo_info.get('display_name', location),
                        "geocoding_success": True
                    }
                else:
                    # No results from geocoding
                    location_details = {
                        "name": location,
                        "country": "Unknown",
                        "state": "",
                        "coordinates": {"lat": 0, "lon": 0},
                        "full_address": location,
                        "geocoding_success": False
                    }
            else:
                # HTTP error from geocoding service
                location_details = {
                    "name": location,
                    "country": "Unknown",
                    "state": "",
                    "coordinates": {"lat": 0, "lon": 0},
                    "full_address": location,
                    "geocoding_success": False
                }
        except Exception as e:
            # Exception during geocoding
            location_details = {
                "name": location,
                "country": "Unknown",
                "state": "",
                "coordinates": {"lat": 0, "lon": 0},
                "full_address": location,
                "geocoding_success": False,
                "geocoding_error": str(e)
            }

        # Step 2: Get weather data using wttr.in (supports many location formats)
        # wttr.in can handle: city names, coordinates, airport codes, landmarks, etc.
        wttr_location = location

        # If we have coordinates, use them for more accuracy
        if location_details["coordinates"]["lat"] != 0:
            lat, lon = location_details["coordinates"]["lat"], location_details["coordinates"]["lon"]
            wttr_location = f"{lat},{lon}"

        wttr_url = f"https://wttr.in/{wttr_location}?format=j1"

        weather_response = requests.get(wttr_url, timeout=15)
        weather_data = weather_response.json()

        if 'error' in weather_data or not weather_data.get('current_condition'):
            return json.dumps({
                "error": f"Weather data not available for '{location}'",
                "suggestion": "Try a more specific location (e.g., 'Paris, France' instead of 'Paris')",
                "status": "error"
            })

        current = weather_data['current_condition'][0]
        nearest_area = weather_data['nearest_area'][0]

        # Step 3: Determine timezone using coordinates and country
        timezone_name = 'UTC'  # Default

        # Enhanced timezone detection
        country_lower = location_details["country"].lower()
        state_lower = location_details["state"].lower()

        # Comprehensive timezone mapping
        timezone_rules = [
            # United States
            ('united states', 'new york', 'America/New_York'),
            ('united states', 'florida', 'America/New_York'),
            ('united states', 'massachusetts', 'America/New_York'),
            ('united states', 'california', 'America/Los_Angeles'),
            ('united states', 'washington', 'America/Los_Angeles'),
            ('united states', 'oregon', 'America/Los_Angeles'),
            ('united states', 'texas', 'America/Chicago'),
            ('united states', 'illinois', 'America/Chicago'),
            ('united states', 'colorado', 'America/Denver'),
            ('united states', 'arizona', 'America/Phoenix'),
            ('united states', 'hawaii', 'Pacific/Honolulu'),
            ('united states', 'alaska', 'America/Anchorage'),
            # Other countries
            ('united kingdom', '', 'Europe/London'),
            ('france', '', 'Europe/Paris'),
            ('germany', '', 'Europe/Berlin'),
            ('japan', '', 'Asia/Tokyo'),
            ('australia', 'new south wales', 'Australia/Sydney'),
            ('australia', 'victoria', 'Australia/Melbourne'),
            ('canada', 'ontario', 'America/Toronto'),
            ('canada', 'quebec', 'America/Toronto'),
            ('canada', 'british columbia', 'America/Vancouver'),
            ('russia', '', 'Europe/Moscow'),
            ('china', '', 'Asia/Shanghai'),
            ('india', '', 'Asia/Kolkata'),
            ('brazil', '', 'America/Sao_Paulo'),
        ]

        # Find matching timezone
        for country_rule, state_rule, tz in timezone_rules:
            if country_rule in country_lower or country_rule in location_details["country"].lower():
                if not state_rule or state_rule in state_lower:
                    timezone_name = tz
                    break

        # Get current time in timezone
        try:
            tz = pytz.timezone(timezone_name)
            current_time = datetime.now(tz)
        except:
            tz = pytz.UTC
            current_time = datetime.now(tz)
            timezone_name = 'UTC'

        # Build comprehensive response
        result = {
            "location": {
                "query": location,
                "name": location_details["name"],
                "country": location_details["country"],
                "state": location_details["state"],
                "full_address": location_details["full_address"],
                "coordinates": location_details["coordinates"],
                "nearest_weather_station": nearest_area['areaName'][0]['value']
            },
            "time_info": {
                "current_time": current_time.strftime("%I:%M %p"),
                "current_time_24h": current_time.strftime("%H:%M"),
                "timezone": timezone_name,
                "timezone_abbr": current_time.strftime("%Z"),
                "utc_offset": current_time.strftime("%z"),
                "date": current_time.strftime("%Y-%m-%d"),
                "day_of_week": current_time.strftime("%A"),
                "month_day": current_time.strftime("%B %d"),
                "iso_datetime": current_time.isoformat()
            },
            "weather": {
                "temperature_f": int(current['temp_F']),
                "feels_like_f": int(current['FeelsLikeF']),
                "humidity_percent": int(current['humidity']),
                "condition": current['weatherDesc'][0]['value'],
                "wind_speed_mph": int(current['windspeedMiles']),
                "visibility_miles": int(current['visibility']),
                "pressure_inches": float(current['pressure']),
                "last_updated": current_time.strftime("%Y-%m-%d %H:%M %Z")
            },
            "api_sources": {
                "geocoding": "Nominatim/OpenStreetMap (Free)",
                "weather": "wttr.in (Free)"
            },
            "status": "success"
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to get location info: {str(e)}",
            "status": "error"
        })

class SimplePromptPersistence:
    """Simple persistence for optimized prompts"""

    def __init__(self, file_path: str = "optimized_prompts.json"):
        self.file_path = file_path

    def save_prompts(self, thread_id: str, prompts: List[str]):
        """Save prompts for a thread"""
        try:
            import json
            data = {}
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    data = json.load(f)

            data[thread_id] = prompts

            with open(self.file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save prompts to {self.file_path}: {e}")

    def load_prompts(self, thread_id: str) -> List[str]:
        """Load prompts for a thread"""
        try:
            import json
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    data = json.load(f)
                    return data.get(thread_id, [])
        except Exception as e:
            print(f"Warning: Could not load prompts from {self.file_path}: {e}")
        return []

class AdaptiveAgent:

    def get_conversation_pretty(self, messages: List[BaseMessage]) -> str:
        """Format conversation messages as a readable string"""
        import json
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"👤 Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        lines.append(f"🤖 AI calls {tool_name} tool")
                        args = tool_call.get('args', {})
                        if isinstance(args, dict) and args:
                            for key, value in args.items():
                                if isinstance(value, str):
                                    if 'code' in key.lower() and '\n' in value:
                                        lines.append(f"   with {key}:")
                                        for line in value.split('\n'):
                                            if line.strip():
                                                lines.append(f"      {line}")
                                    else:
                                        lines.append(f"   with {key}: {value}")
                                else:
                                    lines.append(f"   with {key}: {value}")
                elif msg.content:
                    if msg.content.strip().startswith('{'):
                        try:
                            data = json.loads(msg.content)
                            if data.get('type') == 'function' or 'name' in data:
                                tool_name = data.get('name', 'unknown')
                                lines.append(f"🤖 AI tries to call {tool_name} (malformed)")
                                params = data.get('parameters', {})
                                if isinstance(params, dict):
                                    for key, value in params.items():
                                        if isinstance(value, str) and 'code' in key.lower() and '\n' in value:
                                            lines.append(f"   with {key}:")
                                            for line in value.split('\n'):
                                                if line.strip():
                                                    lines.append(f"      {line}")
                                        else:
                                            lines.append(f"   with {key}: {value}")
                            else:
                                lines.append(f"🤖 AI: {msg.content}")
                        except:
                            lines.append(f"🤖 AI: {msg.content}")
                    else:
                        lines.append(f"🤖 AI: {msg.content}")
            elif isinstance(msg, ToolMessage):
                lines.append(f"🔧 Tool [{msg.name}] output: {msg.content}")
            else:
                msg_type = type(msg).__name__
                lines.append(f"📨 {msg_type}: {getattr(msg, 'content', '[no content]')}")
        return '\n'.join(lines)

    class Evaluation(BaseModel):
        """Boolean evaluation model for all three aspects"""
        response_pass: bool = Field(description="True if response meets the expected criteria, False otherwise")
        response_reasoning: str = Field(description="Reasoning for the response evaluation")
        chain_of_thought_pass: bool = Field(description="True if chain of thought shows good reasoning, False otherwise")
        chain_of_thought_reasoning: str = Field(description="Reasoning for the chain of thought evaluation")
        prompt_pass: bool = Field(description="True if prompt effectively guided the agent, False otherwise")
        prompt_reasoning: str = Field(description="Reasoning for the prompt evaluation")

    class AdaptiveAgentState(TypedDict):
        """State for the adaptive agent"""
        conversation: List[BaseMessage]
        prompts: List[str]
        feedback: str
        task: str
        eval_system_prompt: str
        expected_react_final_response: str
        scores_prompt: List[bool]
        scores_response: List[bool]
        scores_chain_of_thought: List[bool]
        messages_counts: List[int]
        response_reasoning: str
        chain_of_thought_reasoning: str
        prompt_reasoning: str
        max_attempts: int
        attempts: int
        thread_id: str

    def __init__(self, model, tools, checkpointer):
        self.model = model
        self.tools = tools
        self.checkpointer = checkpointer
        self._graph = None
        self.prompt_persistence = SimplePromptPersistence()

    async def tune(self, initial_prompt: str, task: str, eval_system_prompt: str, expected_react_final_response: str,
                   config: Dict[str, Any] = None, max_attempts: int = 1,
                   mode: str = "init"):
        """Tune the agent based on the evaluation scores"""
        thread_id = config.get("configurable", {}).get("thread_id") if config else None
        # Always rebuild graph to ensure latest configuration
        self._graph = self.create_graph()
        if mode == "init":
            state = {
                "conversation": [],
                "prompts": [initial_prompt],
                "task": task,
                "eval_system_prompt": eval_system_prompt,
                "expected_react_final_response": expected_react_final_response,
                "scores_prompt": [],
                "scores_response": [],
                "scores_chain_of_thought": [],
                "messages_counts": [],
                "max_attempts": max_attempts,
                "attempts": 0,
                "thread_id": thread_id
            }
        elif mode == "continue":
            # Load saved prompts from simple persistence
            saved_prompts = self.prompt_persistence.load_prompts(thread_id)
            if saved_prompts:
                state = {
                    "conversation": [],
                    "prompts": saved_prompts,
                    "task": task,
                    "eval_system_prompt": eval_system_prompt,
                    "expected_react_final_response": expected_react_final_response,
                    "scores_prompt": [],
                    "scores_response": [],
                    "scores_chain_of_thought": [],
                    "messages_counts": [],
                    "max_attempts": max_attempts,
                    "attempts": 0,
                    "thread_id": thread_id
                }
            else:
                # Fallback to init mode if no saved prompts found
                print("No saved prompts found, falling back to init mode...")
                state = {
                    "conversation": [],
                    "prompts": [initial_prompt],
                    "task": task,
                    "eval_system_prompt": eval_system_prompt,
                    "expected_react_final_response": expected_react_final_response,
                    "scores_prompt": [],
                    "scores_response": [],
                    "scores_chain_of_thought": [],
                    "messages_counts": [],
                    "max_attempts": max_attempts,
                    "attempts": 0,
                    "thread_id": thread_id
                }
        result = await self._graph.ainvoke(state, config=config)
        return result

    @vero_agent("Running")
    async def run_react_agent(self, state): 
        current_prompt = state["prompts"][-1]
        print(f"[gold1]{current_prompt}[/gold1]")
        executor = create_react_agent(model=self.model, tools=self.tools, prompt=current_prompt)
        messages = [HumanMessage(content=state["task"])]
        result = await executor.ainvoke({
            "messages": messages
        }, config={"recursion_limit": 100})
        last_content = result["messages"][-1].content if isinstance(result, dict) and result.get("messages") else ""
        response_preview = (last_content[:160] + "…") if len(last_content) > 160 else last_content
        print(f"[dim]Attempt: {state['attempts']}/{state['max_attempts']}[/dim]")
        print(f"[dim]Response: {response_preview}[/dim]")
        return {"conversation": result}

    def should_continue(self, state) -> str:
        if state["max_attempts"] == 0:
            return "end"
        elif state["attempts"] >= state["max_attempts"]:
            return "end"
        elif (len(state["scores_response"]) > 0 and
              len(state["scores_chain_of_thought"]) > 0 and
              len(state["scores_prompt"]) > 0 and
              state["scores_response"][-1] and
              state["scores_chain_of_thought"][-1] and
              state["scores_prompt"][-1]):
            return "end"
        else:
            return "evaluate_comprehensive"

    def should_continue_after_eval(self, state) -> str:
        """Route after evaluation: if all tests passed, end; otherwise generate feedback"""
        scores_response = state.get("scores_response", [])
        scores_chain_of_thought = state.get("scores_chain_of_thought", [])
        scores_prompt = state.get("scores_prompt", [])

        # If all tests passed, we're done
        if (len(scores_response) > 0 and
            len(scores_chain_of_thought) > 0 and
            len(scores_prompt) > 0 and
            scores_response[-1] and
            scores_chain_of_thought[-1] and
            scores_prompt[-1]):
            return "end"

        # If any test failed, generate feedback to improve
        return "generate_feedback"

    @vero_agent("Evaluating")
    async def evaluate_comprehensive(self, state):
        conversation = state["conversation"]
        task = state["task"]
        expected_react_final_response = state["expected_react_final_response"]
        current_prompt = state["prompts"][-1]
        eval_system_prompt = state["eval_system_prompt"]
        structured_model = self.model.with_structured_output(self.Evaluation)
        messages = conversation.get("messages", []) if isinstance(conversation, dict) else conversation
        last_message = messages[-1] if messages else None
        agent_response = last_message.content if last_message else "No response"
        full_chain_of_thought = self.get_conversation_pretty(messages)
        message_count = len(messages)
        eval_system = SystemMessage(content=f"""Evaluate the agent's performance across three dimensions with TRUE/FALSE judgments:

1. RESPONSE EVALUATION:
   Task: {task}
   Expected results: {expected_react_final_response}
   Agent's response: {agent_response}

   PASS CRITERIA: Agent's final answer matches or is very close to the expected result.
   Return TRUE if the response is correct or acceptably close, FALSE if significantly wrong.

2. CHAIN OF THOUGHT EVALUATION:
   Agent's full chain of thought:
   {full_chain_of_thought}

   PASS CRITERIA: Agent used logical reasoning, appropriate tools, and showed good problem-solving approach.
   Return TRUE if reasoning was sound and tools were used appropriately, FALSE if reasoning was fundamentally flawed.

3. PROMPT EVALUATION:
   System prompt: {current_prompt}
   Requirements: {eval_system_prompt}

   PASS CRITERIA: Prompt effectively guided the agent to use the right approach and meet the requirements.
   Return TRUE if prompt provided good guidance, FALSE if prompt failed to guide the agent properly.

For each dimension, provide TRUE/FALSE and detailed reasoning explaining your decision.

<output_format>
{self.Evaluation.model_json_schema()}
</output_format>""")
        try:
            eval_human = HumanMessage(content="Please evaluate all three dimensions according to the criteria above.")
            evaluation = await structured_model.ainvoke([eval_system, eval_human])
        except Exception as e:
            evaluation = self.Evaluation(
                response_pass=False,
                response_reasoning=f"Failed to evaluate response: {str(e)}",
                chain_of_thought_pass=False,
                chain_of_thought_reasoning=f"Failed to evaluate chain of thought: {str(e)}",
                prompt_pass=False,
                prompt_reasoning=f"Failed to evaluate prompt: {str(e)}"
            )
        pass_count = sum([evaluation.response_pass, evaluation.chain_of_thought_pass, evaluation.prompt_pass])
        pass_rate = (pass_count / 3) * 100
        response_status = "✅ PASS" if evaluation.response_pass else "❌ FAIL"
        cot_status = "✅ PASS" if evaluation.chain_of_thought_pass else "❌ FAIL"
        prompt_status = "✅ PASS" if evaluation.prompt_pass else "❌ FAIL"
        print(f"[dim]Attempt: {state['attempts']}/{state['max_attempts']}[/dim]")
        print(f"[dim]Response: {response_status} | CoT: {cot_status} | Prompt: {prompt_status} | Pass Rate: {pass_rate:.0f}%[/dim]")
        scores_response = state["scores_response"]
        scores_response.append(evaluation.response_pass)
        scores_chain_of_thought = state.get("scores_chain_of_thought", [])
        scores_chain_of_thought.append(evaluation.chain_of_thought_pass)
        scores_prompt = state["scores_prompt"]
        scores_prompt.append(evaluation.prompt_pass)
        messages_counts = state.get("messages_counts", [])
        messages_counts.append(message_count)
        return {
            "conversation": conversation,
            "scores_response": scores_response,
            "response_reasoning": evaluation.response_reasoning,
            "scores_chain_of_thought": scores_chain_of_thought,
            "chain_of_thought_reasoning": evaluation.chain_of_thought_reasoning,
            "scores_prompt": scores_prompt,
            "prompt_reasoning": evaluation.prompt_reasoning,
            "messages_counts": messages_counts,
            "attempts": state["attempts"] + 1
        }

    @vero_agent("Guiding")
    async def generate_feedback(self, state):
        scores_response = state["scores_response"]
        scores_chain_of_thought = state["scores_chain_of_thought"]
        scores_prompt = state["scores_prompt"]
        chain_of_thought_reasoning = state["chain_of_thought_reasoning"]
        task = state["task"]
        expected_results = state["expected_react_final_response"]
        eval_system_prompt = state["eval_system_prompt"]
        response_reasoning = state.get("response_reasoning", "No response evaluation available")
        prompt_reasoning = state.get("prompt_reasoning", "No prompt evaluation available")
        latest_response_pass = scores_response[-1] if scores_response else False
        latest_cot_pass = scores_chain_of_thought[-1] if scores_chain_of_thought else False
        latest_prompt_pass = scores_prompt[-1] if scores_prompt else False
        response_status = "PASS" if latest_response_pass else "FAIL"
        cot_status = "PASS" if latest_cot_pass else "FAIL"
        prompt_status = "PASS" if latest_prompt_pass else "FAIL"
        feedback_system = SystemMessage(content=f"""You are a generalist AI mentor. Guide the agent to develop transferable problem-solving skills.

Performance: Response {response_status}, Reasoning {cot_status}, Prompt {prompt_status}

Optimization Context:
- Task: {task}
- Expected Result: {expected_results}
- Prompt Requirements: {eval_system_prompt}

Evaluation Context:
- Chain of thought reasoning: {chain_of_thought_reasoning}
- Response evaluation: {response_reasoning}
- Prompt evaluation: {prompt_reasoning}

Provide exactly 3 insights in this format:
🔍 Pattern: [What pattern needs attention?]
🎯 Skill: [What adaptive skill to develop?]
🔄 Action: [What to try differently?]

Keep each insight to 15 words maximum. Focus on meta-cognitive skills, not task-specific solutions.""")
        feedback_human = HumanMessage(content="Based on the agent's current performance and reasoning, what generalist guidance would help it become a better problem-solver across any task domain?")
        feedback_response = await self.model.ainvoke([feedback_system, feedback_human])
        feedback = feedback_response.content
        preview = (feedback[:200] + "…") if len(feedback) > 200 else feedback
        print(f"[dim]Feedback: {preview}[/dim]")
        return {"feedback": feedback}

    @vero_agent("Optimizing")
    async def optimize_prompt(self, state):
        feedback = state["feedback"]
        prompts = state["prompts"]
        current_prompt = prompts[-1]
        eval_system_prompt = state["eval_system_prompt"]
        task = state["task"]
        expected_results = state["expected_react_final_response"]
        chain_of_thought_reasoning = state["chain_of_thought_reasoning"]
        response_reasoning = state["response_reasoning"]
        prompt_reasoning = state["prompt_reasoning"]

        available_tools = []
        if self.tools:
            for tool in self.tools:
                available_tools.append(f"- {tool.name}: {tool.description}")
        tools_list = "\n".join(available_tools)

        # Standard optimization mode
        optimize_system = SystemMessage(content=f"""Current prompt: {current_prompt}

Optimization Context:
- Task: {task}
- Expected Result: {expected_results}
- Requirements: {eval_system_prompt}
- Available tools: {tools_list}

Evaluation Analysis:
- Agent's reasoning process: {chain_of_thought_reasoning}
- Response evaluation: {response_reasoning}
- Prompt evaluation: {prompt_reasoning}

Generalist feedback: {feedback}

Create an improved system prompt that addresses the evaluation insights and feedback. Provide only the prompt, nothing else.""")
        optimize_human = HumanMessage(content="Create the optimized prompt.")

        optimize_response = await self.model.ainvoke([optimize_system, optimize_human])
        new_prompt = optimize_response.content
        prompts.append(new_prompt)

        preview = (new_prompt[:200] + "…") if len(new_prompt) > 200 else new_prompt
        print(f"[dim]New Prompt: {preview}[/dim]")

        # Save prompts to persistence - get thread_id from state if available
        thread_id = state.get("thread_id", "default")
        self.prompt_persistence.save_prompts(thread_id, prompts)

        return {
            "prompts": prompts,
        }

    def create_graph(self):
        graph_builder = StateGraph(self.AdaptiveAgentState)
        graph_builder.add_node("run_react_agent", self.run_react_agent)
        graph_builder.add_node("evaluate_comprehensive", self.evaluate_comprehensive)
        graph_builder.add_node("generate_feedback", self.generate_feedback)
        graph_builder.add_node("optimize_prompt", self.optimize_prompt)
        graph_builder.add_edge(START, "run_react_agent")
        graph_builder.add_conditional_edges("run_react_agent", self.should_continue, {"evaluate_comprehensive": "evaluate_comprehensive", "end": END})
        graph_builder.add_conditional_edges("evaluate_comprehensive", self.should_continue_after_eval, {"generate_feedback": "generate_feedback", "end": END})
        graph_builder.add_edge("generate_feedback", "optimize_prompt")
        graph_builder.add_edge("optimize_prompt", "run_react_agent")
        return graph_builder.compile(checkpointer=self.checkpointer)

if __name__ == "__main__":
    import asyncio
    model = ChatOpenAI(model="gpt-5")
    tools = [get_location_info]
    agent = AdaptiveAgent(model, tools, MemorySaver())
    async def main():
        result = await agent.tune(
            initial_prompt="You are an assistant",
            task="Compare current weather in NYC vs San Francisco and recommend which city is better to be in right now.",
            eval_system_prompt="System prompt requirements: - must list available tools for all code execution. - do not reveal implementation details beforehand - system prompt need to be about 200 words",
            expected_react_final_response="A clean UI-style display with NO CODE OUTPUT, only the final visual results. Should show: (1) Title header, (2) Side-by-side comparison cards for each city showing current temp, conditions, wind, humidity with appropriate emojis, (3) Current timestamp, (4) A final recommendation section with clear winner and data-driven reasoning - all formatted as a visual interface, not code. Must use real data and determine winner based on actual conditions.",
            config={"configurable": {"thread_id": "fresh-start-2025"}, "recursion_limit": 100},
            max_attempts=3,
            mode="continue"
        )
        scores_response = result["scores_response"]
        scores_chain_of_thought = result["scores_chain_of_thought"]
        scores_prompt = result["scores_prompt"]
        messages_counts = result.get("messages_counts", [])
        prompts = result.get("prompts", [])
        print_performance_table(
            prompts=prompts,
            scores_response=scores_response,
            scores_chain_of_thought=scores_chain_of_thought,
            scores_prompt=scores_prompt,
            messages_counts=messages_counts
        )

        # Print the final response from the agent
        conversation = result.get("conversation", {})
        messages = conversation.get("messages", []) if isinstance(conversation, dict) else conversation
        if messages:
            last_message = messages[-1]
            final_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
            console.print("\n[bold cyan]Final Agent Response:[/bold cyan]")
            console.print(f"[white]{final_response}[/white]")
    asyncio.run(main())
