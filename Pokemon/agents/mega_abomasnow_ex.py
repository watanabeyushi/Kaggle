import os
import sys
from collections import defaultdict

from cg.api import AreaType, CardType, Observation, SelectContext, OptionType, Card, Pokemon, all_card_data, to_observation_class

"""
Mega Abomasnow ex Deck
Beginner Friendly
This is a simple deck that attacks with Hammer-lanche.
"""

# Load deck.csv in the dataset
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Fetch card metadata database and create an ID-to-Card lookup table
all_card = all_card_data()
card_table = {c.cardId:c for c in all_card}

# Decklist
Kyogre = 721  # ×2
Snover = 722  # ×4
Mega_Abomasnow_ex = 723  # ×4
Ultra_Ball = 1121  # ×4
Precious_Trolley = 1126  # ×1
Carmine = 1192  # ×4
Lillie_Determination = 1227  # ×2
Surfing_Beach = 1262  # ×2
Basic_Water_Energy = 3  # ×29
Team_Rockets_Petrel = 1219  # ×4
Team_Rockets_Receiver = 1134  # ×4
Crustle = 345  # Blocks damage from ex Pokémon


def is_ex_card(card_id: int) -> bool:
    """Return True if the card is an ex or Mega Evolution ex Pokémon."""
    data = card_table.get(card_id)
    if data == None:
        return card_id == Mega_Abomasnow_ex
    return data.ex or data.megaEx


def blocks_ex_attacks(op_active_id: int) -> bool:
    """Return True if the opponent's Active Pokémon prevents ex attacks."""
    return op_active_id == Crustle


def waiting_evolve(field_counts: dict, hand_counts: dict) -> bool:
    """Snover is ready to evolve into Mega Abomasnow ex this turn."""
    return (
        field_counts[Snover] >= 1
        and field_counts[Mega_Abomasnow_ex] == 0
        and hand_counts[Mega_Abomasnow_ex] >= 1
    )


def assess_hand_state(
    my_state,
    hand_counts: dict,
    field_counts: dict,
    discard_counts: dict,
    ex_attack_blocked: bool,
    bench_attacker_index0: int,
    bench_attacker_index1: int,
) -> tuple[bool, bool, int, bool]:
    """Return should_draw, bad_hand, draw_boost, want_precious_trolley."""
    water_in_hand = hand_counts[Basic_Water_Energy]
    missing_snover = field_counts[Snover] + hand_counts[Snover] == 0
    missing_mega = field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex] == 0
    missing_kyogre = field_counts[Kyogre] + hand_counts[Kyogre] == 0
    mega_attacker_ready = bench_attacker_index0 >= 0

    want_precious = hand_counts[Precious_Trolley] >= 1 and (
        water_in_hand >= 2
        or my_state.handCount >= 5
        or missing_mega
        or missing_snover
        or missing_kyogre
    )
    if hand_counts[Precious_Trolley] == 0 and (missing_mega or missing_snover or missing_kyogre):
        want_precious = True

    bad_hand = (
        water_in_hand >= 3
        or (my_state.handCount >= 5 and water_in_hand >= 2)
        or my_state.handCount >= 6
        or missing_snover
        or (missing_mega and not mega_attacker_ready)
        or (missing_kyogre and (ex_attack_blocked or discard_counts[Basic_Water_Energy] >= 3))
        or want_precious
    )

    draw_boost = 0
    if bad_hand:
        draw_boost += water_in_hand * 400
        draw_boost += max(0, my_state.handCount - 4) * 300
        if missing_mega:
            draw_boost += 800
        if missing_snover:
            draw_boost += 600
        if missing_kyogre:
            draw_boost += 600
        if hand_counts[Precious_Trolley] >= 1:
            draw_boost += 700

    should_draw = bad_hand and not waiting_evolve(field_counts, hand_counts)
    return should_draw, bad_hand, draw_boost, want_precious


def discard_draw_support_count(discard_counts: dict) -> int:
    return (
        discard_counts[Carmine]
        + discard_counts[Lillie_Determination]
        + discard_counts[Team_Rockets_Petrel]
    )


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None

def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return my_deck

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
            
    field_counts = defaultdict(int)  # Number of cards per card ID on the Bench and in the Active Spot
    hand_counts = defaultdict(int)  # Number of cards per card ID in hand
    discard_counts = defaultdict(int)  # Number of cards per card ID in discard pile

    # A Pokémon ready to attack immediately
    bench_attacker_index0 = -1  # Mega Abomasnow ex
    bench_attacker_index1 = -1  # Kyogre
    for i, card in enumerate(my_state.bench):
        field_counts[card.id] += 1
        if card.id == Mega_Abomasnow_ex and len(card.energies) >= 2:
            bench_attacker_index0 = i
        elif card.id == Kyogre and len(card.energies) >= 1:
            bench_attacker_index1 = i

    # Count the number of cards in hand
    for card in my_state.hand:
        hand_counts[card.id] += 1

    # Count the number of cards in discard pile
    for card in my_state.discard:
        discard_counts[card.id] += 1

    op_active_hp = 0
    op_active_id = 0
    for card in state.players[1 - my_index].active:
        if card == None:  # While game setup is in progress
            continue
        op_active_id = card.id
        op_active_hp = card.hp

    active_id = 0
    for card in my_state.active:
        if card == None:
            continue
        active_id = card.id

    ex_attack_blocked = blocks_ex_attacks(op_active_id) and is_ex_card(active_id)

    should_draw, bad_hand, draw_boost, want_precious = assess_hand_state(
        my_state,
        hand_counts,
        field_counts,
        discard_counts,
        ex_attack_blocked,
        bench_attacker_index0,
        bench_attacker_index1,
    )
    discard_support = discard_draw_support_count(discard_counts)
    
    # If opponent HP <= (Basic Water Energy in discard pile * 20), Kyogre can KO.
    prefer_ky = op_active_hp <= 20 * discard_counts[Basic_Water_Energy]
    switch_index = -1
    for card in my_state.active:
        if card == None:  # While game setup is in progress
            continue
        field_counts[card.id] += 1
        if ex_attack_blocked and bench_attacker_index1 >= 0:
            switch_index = bench_attacker_index1  # Kyogre can damage Crustle.
        elif card.id == Mega_Abomasnow_ex and len(card.energies) >= 2:
            if prefer_ky and bench_attacker_index1 >= 0:
                switch_index = bench_attacker_index1  # Switching to Kyogre is preferable.
        elif card.id == Kyogre and len(card.energies) >= 1:
            if not prefer_ky and bench_attacker_index0 >= 0 and not blocks_ex_attacks(op_active_id):
                switch_index = bench_attacker_index0  # Switching to Mega Abomasnow ex is preferable.
        elif bench_attacker_index0 >= 0 and not blocks_ex_attacks(op_active_id):
            switch_index = bench_attacker_index0  # Switching to Mega Abomasnow ex is preferable.

    if ex_attack_blocked and switch_index < 0:
        for i, bench_card in enumerate(my_state.bench):
            if bench_card.id == Kyogre:
                switch_index = i
                break
    
    # Iterate over every possible option and assign a heuristic score.
    scores = []  # Score for each action
    for o in select.option:
        score = 0  # The default and baseline score is 0.
        if o.type == OptionType.NUMBER:
            score = o.number  # e.g., for "draw X cards"
        elif o.type == OptionType.YES:
            score = 1  # Prefer "Yes"
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card != None:
                energy_count = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                if (context == SelectContext.SWITCH
                    or context == SelectContext.TO_ACTIVE
                    or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    # Selection of the Pokémon to send to the Active Spot
                    score += energy_count * 2  # Prioritize Pokémon with Energy attached.
                    if o.index == switch_index:
                        score += 100
                    if blocks_ex_attacks(op_active_id):
                        if card.id == Kyogre:
                            score += 150
                        elif is_ex_card(card.id):
                            score -= 100
                    elif card.id == Mega_Abomasnow_ex:
                        score += 20
                    elif card.id == Kyogre:
                        score += 10
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    # When choosing a card to Bench or add to the hand.
                    if card.id == Snover:
                        if field_counts[card.id] >= 1:
                            score += 5
                        elif field_counts[Mega_Abomasnow_ex] >= 1:
                            score += 15
                        else:
                            score += 30
                    elif card.id == Mega_Abomasnow_ex:
                        if field_counts[Snover] >= 1 and field_counts[card.id] + hand_counts[card.id] == 0:
                            score += 100
                        else:
                            score += 10
                    elif card.id == Kyogre:
                        if field_counts[card.id] >= 1:
                            score += 1
                        else:
                            score += 20
                elif context == SelectContext.DISCARD:
                    # When choosing cards to discard.
                    if card.id == Basic_Water_Energy:
                        score += 100  # Prioritize Basic Water Energy for discard.
                    elif card.id == Mega_Abomasnow_ex:
                        score += 10
                    elif card.id == Carmine:
                        if hand_counts[Lillie_Determination] >= 1:
                            score += 30
                        elif bad_hand:
                            score -= 80
                    elif card.id in (Team_Rockets_Petrel, Team_Rockets_Receiver):
                        if hand_counts[Lillie_Determination] >= 1:
                            score += 30
                        elif bad_hand:
                            score -= 80
                    elif card.id == Precious_Trolley and bad_hand:
                        score -= 100
                    elif card.id == Lillie_Determination:
                        score -= 20

                    if hand_counts[card.id] >= 2:
                        score += 500  # Prioritize discarding duplicate cards.
                    hand_counts[card.id] -= 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            score = 10000
            if card.id == Precious_Trolley:
                if should_draw and hand_counts[Precious_Trolley] >= 1:
                    score = 8800 + draw_boost
                elif want_precious:
                    score = 3500
                else:
                    score = 2500
            elif card.id == Ultra_Ball:
                if hand_counts[Basic_Water_Energy] >= 3 or (my_state.handCount >= 4 and (field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex] == 0 or field_counts[Mega_Abomasnow_ex] + field_counts[Snover] == 0 or field_counts[Kyogre] == 0)):
                    score = 5200 + (400 if should_draw else 0)
                elif should_draw and (field_counts[Snover] + hand_counts[Snover] == 0 or field_counts[Kyogre] + hand_counts[Kyogre] == 0):
                    score = 4800 + draw_boost
                else:
                    score = -1
            elif card.id == Team_Rockets_Receiver:
                if waiting_evolve(field_counts, hand_counts):
                    score = -1
                elif discard_counts[Team_Rockets_Petrel] >= 1:
                    score = 9200 + draw_boost  # Replay Lambda from discard.
                elif discard_support >= 1:
                    score = 8600 + draw_boost  # Replay Lillie / Carmine / Lambda.
                elif should_draw:
                    score = 7200 + draw_boost  # Virtual 8-Lambda line: dig or set up chain.
                elif field_counts[Mega_Abomasnow_ex] + field_counts[Snover] == 0:
                    score = 3200
                else:
                    score = -1
            elif card.id == Team_Rockets_Petrel:
                if waiting_evolve(field_counts, hand_counts):
                    score = -1
                elif should_draw:
                    score = 8000 + draw_boost  # Search supporters from deck (incl. Lambda).
                else:
                    score = 3400
            elif card.id == Lillie_Determination:
                if waiting_evolve(field_counts, hand_counts):
                    score = -1
                elif should_draw:
                    score = 7800 + draw_boost
                else:
                    score = 3600
            elif card.id == Carmine:
                if waiting_evolve(field_counts, hand_counts):
                    score = -1
                elif should_draw:
                    score = 7600 + draw_boost
                else:
                    score = 3400
        elif o.type == OptionType.ATTACH:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 5000
            energy_count = len(pokemon.energies)
            if energy_count == 0:
                if o.inPlayArea == AreaType.BENCH:
                    score += 1
            if blocks_ex_attacks(op_active_id):
                if pokemon.id == Mega_Abomasnow_ex:
                    score -= 400
                elif pokemon.id == Kyogre:
                    score += 300
                    if bench_attacker_index1 < 0 and o.inPlayArea == AreaType.BENCH:
                        score += 200
            if pokemon.id == Snover:
                score += 1
                if energy_count == 1:
                    score -= 100
                elif energy_count >= 2:
                    score -= 400
                if bench_attacker_index0 >= 0:
                    score -= 300
            elif pokemon.id == Mega_Abomasnow_ex:
                score += 10
                if energy_count == 1:
                    score += 30
                elif energy_count >= 2:
                    score -= 300
                if bench_attacker_index0 >= 0:
                    score -= 200
            elif pokemon.id == Kyogre:
                score += 5
                if len(pokemon.energies) >= 1:
                    score -= 200
                if bench_attacker_index1 >= 0:
                    score -= 200
            if o.inPlayArea == AreaType.ACTIVE:
                if bench_attacker_index0 >= 0 and bench_attacker_index1 >= 0 and energy_count <= 2:
                    score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 10000 + len(pokemon.energies)
            if blocks_ex_attacks(op_active_id) and ex_attack_blocked:
                score -= 8000
            elif should_draw and not waiting_evolve(field_counts, hand_counts):
                score -= 1500
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Surfing_Beach and switch_index >= 0:
                score = 2500 if ex_attack_blocked else 2000  # Prioritize over retreating.
            else:
                score = -1
        elif o.type == OptionType.RETREAT:
            if switch_index >= 0:
                score = 2000 if ex_attack_blocked else 1500
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            if ex_attack_blocked:
                score = -1
            else:
                score = 1000
                if o.attackId == 1042:  # Riptide
                    score += discard_counts[Basic_Water_Energy] * 20 - 90
                elif o.attackId == 1046:  # Hammer-lanche
                    if op_active_hp <= 200:
                        score -= 100
                    else:
                        score += 100

        scores.append(score)

    # Select in descending order of score
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
