import json
import os
import openai  # or your GPT-4o library
from tqdm import tqdm

# ---------------
# Adjust these based on your GPT-4o environment:
# ---------------
# openai.api_key = "<YOUR_API_KEY>"
# openai.api_base = "https://api.gpt4o.com/v1"

def generate_label_gpt_4o(object_text):
    """
    Calls GPT-4o (or other LLM) to generate a natural textual label
    for the segmentation object.

    :param object_text: The raw text describing the class or object
    :return: A short descriptive label or sentence about that object
    """

    # Example system prompt
    system_prompt = (
        "You are a helpful assistant that generates short, natural-sounding labels "
        "for image segmentation tasks. The user will provide the name or short description "
        "of the object to be segmented. Respond with a single sentence or phrase that "
        "describes the segmentation target as if explaining it in natural language."
    )

    # Example user prompt
    user_prompt = f"Object to segment: {object_text}\nGenerate a short descriptive label:"

    try:
        # This is an example usage. Adapt to your GPT-4o method signature.
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or whichever GPT-4o model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=60,
        )
        label = response["choices"][0]["message"]["content"].strip()
        return label
    except Exception as e:
        print(f"GPT-4o error: {e}")
        return "Segmentation label placeholder"


def main(base_image_dir):
    """
    Example script that:
    1. Reads your reason_seg metadata JSON files
    2. Generates natural-sounding labels for each object
    3. Stores the output in a new JSON
    """
    # Path to your reason_seg annotation JSONs (example)
    # We'll assume your original code references something like:
    #    base_image_dir/reason_seg/<train|val|...>/*.json
    # You might need to adapt the path to match your actual environment.

    reasonseg_dir = os.path.join(base_image_dir, "reason_seg", "ReasonSeg", "train")

    output_records = []
    for filename in os.listdir(reasonseg_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(reasonseg_dir, filename)
            with open(json_path, "r") as f:
                data = json.load(f)  # structure of your existing .json

            # Suppose each JSON has a list of "objects" or "annotations".
            # The function `get_mask_from_json` in your dataset code shows 
            # that it can produce multiple text categories `sents`.
            # We'll emulate that logic: 
            #   data["objects"] => a list of object labels (?)
            # This is just a rough example; adapt to your real structure:

            if "objects" in data:  # or "sents" or however your JSON is laid out
                new_objects = []
                for obj in data["objects"]:
                    # object_text is the raw class name, e.g. "bowling ball?"
                    # We'll remove punctuation like "?"
                    cleaned = obj.lower().replace("?", "")
                    natural_label = generate_label_gpt_4o(cleaned)
                    new_objects.append({
                        "original_text": obj,
                        "natural_label": natural_label
                    })
                # Store or update the data
                output_records.append({
                    "image": data.get("image", filename.replace(".json", ".jpg")),
                    "objects": new_objects
                })

    # Write out a combined JSON that has the natural labels
    out_path = os.path.join(base_image_dir, "reason_seg_labels_gpt4o.json")
    with open(out_path, "w") as f:
        json.dump(output_records, f, indent=2)

    print(f"Done! GPT-4o-labeled data saved to {out_path}")


if __name__ == "__main__":
    # For demonstration:
    base_dir = "/path/to/your/base_image_dir"
    main(base_dir)
