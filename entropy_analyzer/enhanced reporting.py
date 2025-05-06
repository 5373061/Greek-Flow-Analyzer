# In the report generation section:

# Add entropy metrics to the report
entropy_section = ""
if entropy_data:
    energy_state = entropy_data.get("energy_state", {})
    entropy_section += f"\nENERGY STATE ANALYSIS:\n"
    entropy_section += f"State: {energy_state.get('state', 'N/A')}\n"
    entropy_section += f"Direction: {energy_state.get('direction', 'N/A')}\n"
    entropy_section += f"Average Normalized Entropy: {energy_state.get('average_normalized_entropy', 'N/A'):.2f}\n"
    
    # Add anomalies if present
    anomalies = entropy_data.get("anomalies", {}).get("anomalies", {})
    if anomalies:
        entropy_section += f"\nANOMALIES DETECTED ({len(anomalies)}):\n"
        for metric, description in anomalies.items():
            entropy_section += f"- {metric}: {description}\n"

# Append to report file
if entropy_section:
    with open(report_file, "a") as f:
        f.write(entropy_section)