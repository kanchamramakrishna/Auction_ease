def matches_role(role, required_role):
    role = str(role).lower()

    if required_role == "Any":
        return True

    if required_role == "Wicketkeeper":
        return "wicketkeeper" in role

    if required_role == "Batter":
        return (
            "batter" in role
            and "wicketkeeper" not in role
            and "all-rounder" not in role
            and "all rounder" not in role
            and "bowler" not in role
        )

    if required_role == "Bowler":
        return (
            "bowler" in role
            and "all-rounder" not in role
            and "all rounder" not in role
            and "wicketkeeper" not in role
        )

    if required_role == "All-Rounder":
        return (
            "all-rounder" in role
            or "all rounder" in role
        )

    return False


def matches_bowling_type(role, bowling_type):
    role = str(role).lower()

    if bowling_type == "Any":
        return True

    spin_terms = [
        "spin",
        "off-break",
        "leg-break",
        "orthodox",
        "chinaman"
    ]

    fast_terms = [
        "fast",
        "medium",
        "pace",
        "seam"
    ]

    if bowling_type == "Spin":
        return any(term in role for term in spin_terms)

    if bowling_type == "Fast":
        return any(term in role for term in fast_terms)

    return False


def filter_players(data, requirements):
    required_role = requirements["required_role"]
    bowling_type = requirements.get("bowling_type", "Any")

    eligible = data[
        data["role"].apply(
            lambda role: matches_role(role, required_role)
        )
    ].copy()

    if required_role in ["Bowler", "All-Rounder"]:
        eligible = eligible[
            eligible["role"].apply(
                lambda role: matches_bowling_type(
                    role, bowling_type
                )
            )
        ].copy()

    return eligible.drop_duplicates(subset=["player"]).copy()