import os
from collections import defaultdict

def average_luminosity_by_hour(folder_path):
    """
    Reads .lum files named as:
    YYYY_MM_DD__HH_MM_SS__LUMINOSITY.lum

    Returns a dict: {hour (0-23): average_luminosity}
    """
    luminosity_by_hour = defaultdict(list)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".lum"):
            continue

        try:
            # Remove extension
            name = filename[:-4]

            # Split parts
            date_part, time_part, luminosity_part = name.split("__")

            # Extract hour
            hour = int(time_part.split("_")[0])

            # Extract luminosity
            luminosity = float(luminosity_part)

            luminosity_by_hour[hour].append(luminosity)

        except (ValueError, IndexError):
            # Skip files that don't match the expected format
            print(f"Skipping invalid filename: {filename}")

    # Compute averages
    average_by_hour = {
        hour: sum(values) / len(values)
        for hour, values in luminosity_by_hour.items()
    }

    return average_by_hour


if __name__ == "__main__":
    folder = "/home/xav/Documents/luminosity"  # <-- change this

    averages = average_luminosity_by_hour(folder)

    print("Average Luminosity by Hour of Day:")
    for hour in sorted(averages):
        print(f"{hour:02d}:00 - {averages[hour]:.2f}")
