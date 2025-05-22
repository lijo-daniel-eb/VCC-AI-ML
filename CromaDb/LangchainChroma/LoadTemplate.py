import json
import re

# Load the template from template.json
template_file = "RiskEventTemplate.json"
with open(template_file, "r") as file:
    template = json.load(file)

# Load the audit data from auditData.json
audit_data_file = "auditData.json"
with open(audit_data_file, "r") as file:
    audit_data = json.load(file)

# Populate the template with values from auditData.json
populated_texts = []
pattern = re.compile(r"{(.*?)}")
for record in audit_data:
    keys = pattern.findall(template["text"])
    populated_text = template["text"]
    for key in keys:
        populated_text = populated_text.replace(f"{{{key}}}", str(record.get(key, "")))
    # If ExtendedPropertiesAccess exists and is a dict, append its properties
    if "ExtendedPropertiesAccess" in record and isinstance(record["ExtendedPropertiesAccess"], dict):
        ext_props = record["ExtendedPropertiesAccess"]
        ext_props_str = " ".join(f"{k}: {v}" for k, v in ext_props.items())
        populated_text += " " + ext_props_str
    populated_texts.append(populated_text)

# Print the populated texts
for text in populated_texts:
    print(text)
