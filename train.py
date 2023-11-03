
import pprint
import sys

import settings

from src.train_model import train_model


if __name__ == "__main__":
    settings_nr = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"\nSettings number: {settings_nr}")

    chosen_settings = settings.get_settings(settings_nr)
    print(f"\nSettings:")
    pprint.pprint(chosen_settings.as_dict)
    print()

    train_model(chosen_settings)
