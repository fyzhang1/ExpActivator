import json
from collections import defaultdict

from utils import eval_action, compute_CSR


def evaluate_json(input_path, width=1080, height=2400):
    groups = defaultdict(list)
    new_data = []

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    type_acc, sr_acc = 0, 0
    results = []

    for item in data:
        label = item.get("answer")
        output = item.get("output")

        # Step 1: Extract structured fields from model output
        action_type, action_point, action_app, action_direction = None, None, None, None

        if isinstance(output, dict):
            action_type = output.get("action_type", "")
            action_point = output.get("point", "")
            action_app = output.get("app_name", "")
            action_direction = output.get("direction", "")
        elif isinstance(output, str):
            action_type = output
        else:
            action_type = ""

        # Step 2: Build canonical action string based on type
        if action_type == "click" and action_point:
            try:
                x, y = action_point[0], action_point[1]
                action = f"click(coordinates=({x}, {y}))"
            except Exception:
                action = "click"
        elif action_type == "scroll" and action_direction:
            action = f"scroll(direction='{action_direction}')"
        elif action_type == "open_app" and action_app:
            action = f"open_app(app_name='{action_app}')"
        elif action_type:
            action = action_type
        else:
            action = ""

        # Step 3: Evaluate against label
        type_, sr = eval_action(action, label, width, height)
        type_acc += type_
        sr_acc += sr
        item['result'] = f"Type: {type_}, SR: {sr}"

        eid = item.get("id")
        if eid is None:
            continue
        prefix = "_".join(eid.split("_")[:-1])
        groups[prefix].append(item)


    # Re-index episodes and write sorted output for calculating CSR
    for prefix, items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda x: int(x["id"].split("_")[-1])
        )
        for new_idx, item in enumerate(items_sorted):
            item["id"] = f"{prefix}_{new_idx}"
            new_data.append(item)
    CSR = compute_CSR(new_data)

    print(f"Type Accuracy: {type_acc / len(data):.4f}")
    print(f"Success Rate: {sr_acc / len(data):.4f}")
    print(f"Cumulative Success Rate (CSR): {CSR:.4f}")

    # output_path = '.'.join(input_path.split('.')[:-1]) + '_sorted.json'
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Total episodes after reindex: {len(new_data)}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, required=True,
        help="Path to the evaluation result JSON file"
    )
    args = parser.parse_args()
    evaluate_json(args.input_path)