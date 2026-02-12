import os
import pandas as pd
import numpy as np


def check_slip(csv_path, fracture_id, threshold=1e-8):
    """
    Check if a fracture has slipped by reading the displacement jump data.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing displacement jump data
    fracture_id : str
        The fracture identifier (e.g., "Fracture 1")
    threshold : float
        Threshold for determining slip (values above this are considered slip)

    Returns:
    --------
    bool : True if slip occurred, False otherwise
    """
    if not os.path.exists(csv_path):
        return None  # File doesn't exist

    try:
        df = pd.read_csv(csv_path)
        # Get the last time step for the specified fracture
        fracture_data = df[df['fracture_id'] == fracture_id]
        if len(fracture_data) == 0:
            return None

        last_displacement = fracture_data.iloc[-1]['displacement_jump']
        return abs(last_displacement) > threshold
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


if __name__ == "__main__":

    fracture_strike_angles = [45, 40, 35]
    fracture_strike_angles = [45, 43, 41, 39]
    angle_indices = [
        0,  # Injection fracture
        1,  # Central fracture
        2,  # Production fracture
    ]

    # Dictionary to store slip status: {strike: {angle_index: {fracture_num: {"with_wells": bool, "without_wells": bool}}}}
    slip_status = {}

    # Read data for all combinations
    for strike in fracture_strike_angles:
        slip_status[strike] = {}

        for angle_index in angle_indices:
            # Map angle_index to fracture number (1-indexed)
            fracture_num = angle_index + 1
            fracture_id = f"Fracture {fracture_num}"

            slip_status[strike][angle_index] = {"with_wells": None, "without_wells": None}

            # Check with wells
            suffix = f"_strike_{int(strike)}_tilted_fracture_{angle_index}"
            csv_with_wells = f"case_II_with_wells{suffix}_saved_data/well_monitoring/example_7_fractures.csv"
            slip_with = check_slip(csv_with_wells, fracture_id)
            slip_status[strike][angle_index]["with_wells"] = slip_with

            # Check without wells
            csv_without_wells = f"case_II_without_wells{suffix}_saved_data/well_monitoring/example_7_fractures.csv"
            slip_without = check_slip(csv_without_wells, fracture_id)
            slip_status[strike][angle_index]["without_wells"] = slip_without

            if slip_with is not None or slip_without is not None:
                print(f"Strike {strike}°, Angle Index {angle_index} (Fracture {fracture_num}):")
                print(f"  With wells: {slip_with}")
                print(f"  Without wells: {slip_without}")

    # Special handling for strike 45°: Use angle index 0 data for all columns
    # (all angle indices are equivalent due to symmetry for 45° strike)
    print("\n" + "="*60)
    print("Special handling for 45° strike (using angle index 0 data for all)")
    print("="*60)
    if 45 in slip_status:
        reference_data = slip_status[45][0]  # Use angle index 0 as reference
        for angle_index in angle_indices:
            slip_status[45][angle_index] = {
                "with_wells": reference_data["with_wells"],
                "without_wells": reference_data["without_wells"]
            }
            print(f"Strike 45°, Angle Index {angle_index}: Using data from angle index 0")
            print(f"  With wells: {reference_data['with_wells']}")
            print(f"  Without wells: {reference_data['without_wells']}")

    # Assert that there's no case of "slip only without wells"
    print("\n" + "="*60)
    print("Checking for invalid combinations (slip only without wells)...")
    print("="*60)
    for strike in fracture_strike_angles:
        for angle_index in angle_indices:
            with_wells = slip_status[strike][angle_index]["with_wells"]
            without_wells = slip_status[strike][angle_index]["without_wells"]

            # Skip if data is missing
            if with_wells is None or without_wells is None:
                continue

            # Check for invalid combination
            if without_wells and not with_wells:
                raise AssertionError(
                    f"Invalid combination found: Strike {strike}°, Angle Index {angle_index} - "
                    f"Slip without wells but not with wells!"
                )

    print("✓ No invalid combinations found (slip only without wells does not occur)")

    # Create color-coded table
    print("\n" + "="*60)
    print("SLIP SUMMARY TABLE")
    print("="*60)
    print("\nColor coding:")
    print("  🟢 Green:  Slip both with and without wells")
    print("  🟡 Yellow: Slip only with wells")
    print("  🔴 Red:    No slip")
    print("  ⚪ White:  Missing data")
    print("\n")

    # Create the table header
    header = "Strike° | " + " | ".join([f"Angle Index {i}" for i in angle_indices]) + " |"
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

    for strike in fracture_strike_angles:
        row = f"  {strike:2d}    | "
        cells = []

        for angle_index in angle_indices:
            with_wells = slip_status[strike][angle_index]["with_wells"]
            without_wells = slip_status[strike][angle_index]["without_wells"]

            # Determine color
            if with_wells is None or without_wells is None:
                # Missing data
                color = WHITE
                symbol = "⚪"
                cell_text = f"{color}{symbol:^14}{RESET}"
            elif with_wells and without_wells:
                # Green: slip both with and without wells
                color = GREEN
                symbol = "🟢"
                cell_text = f"{color}{symbol:^14}{RESET}"
            elif with_wells and not without_wells:
                # Yellow: slip only with wells
                color = YELLOW
                symbol = "🟡"
                cell_text = f"{color}{symbol:^14}{RESET}"
            else:
                # Red: no slip
                color = RED
                symbol = "🔴"
                cell_text = f"{color}{symbol:^14}{RESET}"

            cells.append(cell_text)

        row += " | ".join(cells) + " |"
        print(row)

    print(separator)

    # Also create an HTML table for better visualization
    print("\n" + "="*60)
    print("HTML TABLE (for documentation)")
    print("="*60)

    html = """
<table border="1" style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr style="background-color: #f0f0f0;">
      <th style="padding: 10px;">Strike (°)</th>
"""

    for angle_index in angle_indices:
        html += f'      <th style="padding: 10px;">Angle Index {angle_index}</th>\n'

    html += """    </tr>
  </thead>
  <tbody>
"""

    for strike in fracture_strike_angles:
        html += f'    <tr>\n      <td style="padding: 10px; font-weight: bold;">{strike}</td>\n'

        for angle_index in angle_indices:
            with_wells = slip_status[strike][angle_index]["with_wells"]
            without_wells = slip_status[strike][angle_index]["without_wells"]

            # Determine color
            if with_wells is None or without_wells is None:
                # Missing data
                bg_color = "#ffffff"
                text = "Missing"
            elif with_wells and without_wells:
                # Green: slip both with and without wells
                bg_color = "#90EE90"
                text = "Both"
            elif with_wells and not without_wells:
                # Yellow: slip only with wells
                bg_color = "#FFFF66"
                text = "Wells Only"
            else:
                # Red: no slip
                bg_color = "#FF6B6B"
                text = "No Slip"

            html += f'      <td style="padding: 10px; background-color: {bg_color};">{text}</td>\n'

        html += '    </tr>\n'

    html += """  </tbody>
</table>
"""

    print(html)

    # Save HTML table to file
    with open("slip_summary_table.html", "w") as f:
        f.write("<html><head><title>Slip Summary Table</title></head><body>\n")
        f.write("<h1>Fracture Slip Summary</h1>\n")
        f.write("<p><strong>Legend:</strong></p>\n")
        f.write("<ul>\n")
        f.write('  <li><span style="background-color: #90EE90; padding: 2px 10px;">Green</span>: Slip both with and without wells</li>\n')
        f.write('  <li><span style="background-color: #FFFF66; padding: 2px 10px;">Yellow</span>: Slip only with wells</li>\n')
        f.write('  <li><span style="background-color: #FF6B6B; padding: 2px 10px;">Red</span>: No slip</li>\n')
        f.write('  <li><span style="background-color: #ffffff; padding: 2px 10px; border: 1px solid #ccc;">White</span>: Missing data</li>\n')
        f.write("</ul>\n")
        f.write(html)
        f.write("</body></html>")

    print("\n✓ HTML table saved to 'slip_summary_table.html'")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
