import pandas as pd
from difflib import SequenceMatcher
import re
import json
import numpy as np

USER_PROFILE_PATH = './samples/user_profile.csv'

def load_user_profile(user_id):
    """Load user profile from CSV and return a formatted description string."""
    df = pd.read_csv(USER_PROFILE_PATH)
    row = df[df['user_id'] == int(user_id)]

    if not row.empty:
        result = row.iloc[0, 1:].tolist()
        text = f'{result[0]}, age: {str(result[1])}, job: {result[2]}, living: {result[3]}, {result[4]}, use: {result[5]}'
        return text
    else:
        print("cannot find user_id")
        return None


def get_direction(coord1, coord2):
    """Determine swipe direction (up/down/left/right) from two coordinates."""
    x1, y1 = coord1
    x2, y2 = coord2
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def text_match(action, label):
    """Check if two action strings are similar enough (threshold=0.7)."""
    similarity = SequenceMatcher(None, action, label).ratio()
    return 1 if similarity >= 0.7 else 0


def convert_format(action):
    """Convert click/long_click action string to box-format representation."""
    if action.startswith("click"):
        action = re.sub(
            r"click\(coordinates=\((\d+),\s*(\d+)\).*",
            r"click(start_box='<|box_start|>(\1,\2)<|box_end|>')",
            action
        )
    elif action.startswith("long_click"):
        action = re.sub(
            r"long_click\(coordinates=\((\d+),\s*(\d+)\).*",
            r"long_click(start_box='<|box_start|>(\1,\2)<|box_end|>')",
            action
        )
    return action


def convert_action_json(action):
    """Convert an action string to JSON format compatible with mobile_use tool calls."""
    result = {"name": "mobile_use", "arguments": {}}

    if action.startswith("click"):
        coord_match = re.search(r"coordinates=\((\d+),\s*(\d+)\)", action)
        result["arguments"]["action"] = "click"
        if coord_match:
            result["arguments"]["coordinate"] = [
                int(coord_match.group(1)), int(coord_match.group(2))
            ]

    if action.startswith("long_click"):
        coord_match = re.search(r"coordinates=\((\d+),\s*(\d+)\)", action)
        result["arguments"]["action"] = "long_click"
        if coord_match:
            result["arguments"]["coordinate"] = [
                int(coord_match.group(1)), int(coord_match.group(2))
            ]

    elif action.startswith("scroll"):
        coord_match = re.search(r"coordinates=\((\d+),\s*(\d+)\)", action)
        dir_match = re.search(r"direction='([^']+)'", action)
        result["arguments"]["action"] = "swipe"
        if dir_match:
            result["arguments"]["direction"] = dir_match.group(1)

    elif action.startswith("type"):
        text_match_result = re.search(r"text='([^']+)'", action)
        result["arguments"]["action"] = "type"
        if text_match_result:
            result["arguments"]["text"] = text_match_result.group(1)

    else:
        match = re.match(r"(\w+)\s*\(", action)
        if match:
            result["arguments"]["action"] = match.group(1)
        else:
            result["arguments"]["action"] = action.strip()

    return json.dumps(result, ensure_ascii=False)


def rescale_coordinates(point, width, height):
    """Rescale coordinates from 0-999 normalized space to actual screen resolution."""
    return [round(point[0] / 999 * width), round(point[1] / 999 * height)]


def convert_action_qwen(action, label, width, height, coord_map=True):
    """Convert qwen-like output (dict) to a standard action string and evaluate."""
    act_type = action.get('action', '')

    if act_type == 'click':
        coord = action.get('coordinate', [0, 0])
        if coord_map:
            coord = rescale_coordinates(coord, width, height)
        action = f"click(coordinates=({coord[0]},{coord[1]}))"

    elif act_type == 'long_click':
        coord = action.get('coordinate', [0, 0])
        if coord_map:
            coord = rescale_coordinates(coord, width, height)
        action = f"long_click(coordinates=({coord[0]},{coord[1]}))"

    elif act_type == 'scroll':
        coord1 = action.get('coordinate', [0, 0])
        coord2 = action.get('coordinate2', [0, 0])
        direction = get_direction(coord1, coord2)
        action = f"scroll(direction='{direction}')"

    elif act_type in ('open_app', 'open'):
        app_name = action.get('app_name') or action.get('text', '')
        action = f"open_app(app_name='{app_name}')"

    elif act_type in ('press_home', 'press_home()', 'press_back', 'press_back()',
                       'wait', 'wait()'):
        act_clean = act_type.replace("()", "")
        action = f"{act_clean}()"

    elif act_type == 'terminate':
        action = 'finished'

    elif act_type == 'swipe':
        coord1 = action.get('coordinate', [0, 0])
        coord2 = action.get('coordinate2', [0, 0])
        direction = get_direction(coord1, coord2)
        action = f"scroll(direction='{direction}')"

    elif act_type == 'type':
        text = action.get('text', '')
        action = f"type(text='{text}')"

    else:
        action = "Error"

    return eval_action(action, label, width, height)


def convert_action_api(action, label, width, height):
    """Convert API model output (str/dict) to a standard action string and evaluate."""
    if isinstance(action, dict):
        try:
            action = json.dumps(action, ensure_ascii=False)
        except Exception:
            action = str(action)
    elif not isinstance(action, str):
        action = str(action)
    action = action.strip()

    if "click" in action:
        coord_match = re.search(r"(\d+)\s+(\d+)", action)
        if coord_match:
            x, y = map(int, coord_match.groups())
            point_convert = rescale_coordinates([x, y], width, height)
            if "long_click" in action:
                action = f"long_click(coordinates=({point_convert[0]},{point_convert[1]}))"
            else:
                action = f"click(coordinates=({point_convert[0]},{point_convert[1]}))"
        else:
            action = "click()"

    elif "open_app" in action:
        app_match = re.search(
            r"app_name[=:]['\"]?([\w\u4e00-\u9fa5A-Za-z0-9_]+)['\"]?", action
        )
        app_name = app_match.group(1) if app_match else ""
        action = f"open_app(app_name='{app_name}')"

    elif "scroll" in action or "swipe" in action:
        direction_match = re.search(r"direction[=:]['\"]?(\w+)['\"]?", action)
        direction = direction_match.group(1) if direction_match else "undefined"
        action = f"scroll(direction='{direction}')"

    elif re.search(r"(press_home|press_back|wait)", action):
        act_type = re.search(r"(press_home|press_back|wait)", action).group(1)
        action = f"{act_type}()"

    elif "terminate" in action or "finished" in action:
        action = "finished"

    elif "type" in action:
        text_match_result = re.search(r"text[=:]['\"](.*?)['\"]", action)
        text = text_match_result.group(1) if text_match_result else ""
        action = f"type(text='{text}')"

    else:
        act_match = re.match(r"(\w+)", action)
        act_type = act_match.group(1) if act_match else "unknown"
        action = f"{act_type}()"

    print('function:')
    print(action, label)
    return eval_action(action, label, width, height)


def eval_action(action, label, width, height):
    """Evaluate a single predicted action against the ground truth label."""
    if action is None:
        return 0, 0

    if label == "finished()":
        if 'finished' in action:
            return 1, 1
        else:
            return 0, 0

    if label in ("press_back()", "press_home()", "wait()"):
        if label == action:
            return 1, 1

    elif label.startswith("open_app"):
        if not action.startswith("open_app"):
            return 0, 0
        label_match = re.search(r"app_name='(.*?)'", label)
        action_match = re.search(r"app_name='(.*?)'", action)
        if not label_match or not action_match:
            return 1, 0
        label_app = label_match.group(1)
        action_app = action_match.group(1)
        if label_app in action_app or action_app in label_app:
            return 1, 1
        else:
            return 1, 0

    elif label.startswith("scroll"):
        if not action.startswith("scroll"):
            return 0, 0
        label_dir_match = re.search(r"direction='(.*?)'", label)
        direction_label = label_dir_match.group(1)
        action_dir_match = re.search(r"direction='(.*?)'", action)
        try:
            direction_action = action_dir_match.group(1)
        except AttributeError:
            direction_action = None
        if direction_label == direction_action:
            return 1, 1
        else:
            return 1, 0

    elif label.startswith("click") or label.startswith("long_click"):
        if not action.startswith("click") and not action.startswith("long_click"):
            return 0, 0
        label_match = re.search(r"\((\d+),\s*(\d+)\)", label)
        action_match = re.search(r'\((\d+),(\d+)\)', action)
        if not label_match or not action_match:
            return 1, 0
        # Normalize both coordinates to 0-1000 space for comparison
        x1, y1 = map(int, label_match.groups())
        x1 = int(x1 / width * 1000)
        y1 = int(y1 / height * 1000)
        x2, y2 = map(int, action_match.groups())
        x2 = int(x2 / width * 1000)
        y2 = int(y2 / height * 1000)
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        relative_distance = distance / 1000
        if relative_distance < 0.14:
            return 1, 1
        else:
            return 1, 0

    elif label.startswith("type"):
        if not action.startswith("type"):
            return 0, 0
        elif text_match(action, label) == 1:
            return 1, 1
        else:
            return 1, 0

    return 0, 0

# calculate CSR
def make_weights(L, mode='exp', alpha=0.6):
    if mode == 'linear':
        w = np.array([L - k for k in range(L)]) + 1.0
    elif mode == 'exp':
        raw = np.exp(-alpha * np.arange(L))
        w = raw / raw.sum() * L
    return w

def compute_CSR(data, mode='exp', alpha=0.6):
    import re
    import numpy as np
    from collections import defaultdict

    episodes = defaultdict(list)
    for item in data:
        ep_full = item["episode_id"]
        match = re.match(r"(.+?)_(\d+)$", ep_full)
        if match:
            ep_id, step_idx = match.groups()
            step_idx = int(step_idx)
        else:
            ep_id = ep_full
            step_idx = 0

        result = item["result"]
        sr_match = re.search(r"SR:\s*(\d+)", result)
        sr = int(sr_match.group(1)) if sr_match else 0

        episodes[ep_id].append((step_idx, sr))

    total_correct = 0.0
    total_weight = 0.0

    for ep_id, steps in episodes.items():
        steps = sorted(steps, key=lambda x: x[0])
        L = len(steps)
        weights = make_weights(L, mode=mode, alpha=alpha)
        success_flags = np.array([sr for _, sr in steps], dtype=float)

        total_correct += np.dot(success_flags, weights)
        total_weight += weights.sum()

    CSR = total_correct / total_weight
    print(f"Cumulative Success Rate: {CSR:.4f}") # we change the name from CER to CSR, which is more intuitive and misunderstanding
    return CSR