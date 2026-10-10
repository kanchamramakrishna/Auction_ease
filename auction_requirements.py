def get_auction_requirements():
    print("\n===== AUCTION EASE =====")
    print("Select player role:")
    print("1. Batter")
    print("2. Wicketkeeper")
    print("3. Bowler")
    print("4. All-Rounder")
    print("5. Any role")

    role_options = {
        "1": "Batter",
        "2": "Wicketkeeper",
        "3": "Bowler",
        "4": "All-Rounder",
        "5": "Any"
    }

    while True:
        choice = input("Enter choice (1-5): ").strip()
        if choice in role_options:
            required_role = role_options[choice]
            break
        print("Invalid choice. Try again.")

    bowling_type = "Any"

    if required_role in ["Bowler", "All-Rounder"]:
        print("\nSelect bowling specialization:")
        print("1. Fast")
        print("2. Spin")
        print("3. Any")

        bowling_options = {
            "1": "Fast",
            "2": "Spin",
            "3": "Any"
        }

        while True:
            choice = input("Enter choice (1-3): ").strip()
            if choice in bowling_options:
                bowling_type = bowling_options[choice]
                break
            print("Invalid choice. Try again.")

    print("\nSelect performance priority:")
    print("1. Batting")
    print("2. Bowling")
    print("3. Balanced")

    priority_options = {
        "1": "Batting",
        "2": "Bowling",
        "3": "Balanced"
    }

    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice in priority_options:
            priority = priority_options[choice]
            break
        print("Invalid choice. Try again.")

    return {
        "required_role": required_role,
        "bowling_type": bowling_type,
        "priority": priority
    }
