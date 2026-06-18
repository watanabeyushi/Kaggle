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


def can_discard_for_ultra_ball(
    hand_counts: defaultdict,
    my_state,
    field_counts: defaultdict,
) -> bool:
    """Return True if the hand can afford Ultra Ball's discard cost."""
    return hand_counts[Basic_Water_Energy] >= 3 or (
        my_state.handCount >= 4
        and (
            field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex] == 0
            or field_counts[Snover] == 0
            or field_counts[Kyogre] + hand_counts[Kyogre] == 0
        )
    )


def want_surfing_beach_for_kyogre_switch(
    active_id: int,
    switch_index: int,
    bench_attacker_index1: int,
    stadium_id: int,
) -> bool:
    """Prefer Surfing Beach when retreating from Mega to bench Kyogre."""
    if stadium_id != 0 and stadium_id != Surfing_Beach:
        return False
    return (
        active_id == Mega_Abomasnow_ex
        and bench_attacker_index1 >= 0
        and switch_index >= 0
    )


def supporter_search_score(
    card_id: int,
    my_state,
    field_counts: defaultdict,
    hand_counts: defaultdict,
    active_id: int,
    switch_index: int,
    bench_attacker_index1: int,
    stadium_id: int,
) -> int | None:
    """Score supporters for Petrel deck search / Receiver discard replay.

    Priority baseline: Trolley > Ultra Ball > Lillie > Carmine > Surfing Beach.
    """
    handled = {
        Precious_Trolley,
        Ultra_Ball,
        Lillie_Determination,
        Carmine,
        Surfing_Beach,
        Team_Rockets_Petrel,
        Team_Rockets_Receiver,
    }
    if card_id not in handled:
        return None

    bench_not_full = len(my_state.bench) < 5
    snover_waiting = field_counts[Snover] >= 1 and field_counts[Mega_Abomasnow_ex] == 0
    mega_outside_field = field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex] == 0

    if card_id == Precious_Trolley:
        return 1000 if bench_not_full else 500
    if card_id == Ultra_Ball:
        if (
            snover_waiting
            and mega_outside_field
            and can_discard_for_ultra_ball(hand_counts, my_state, field_counts)
        ):
            return 850 if bench_not_full else 800
        return 400
    if card_id == Lillie_Determination:
        return 300
    if card_id == Carmine:
        return 200
    if card_id == Surfing_Beach:
        if want_surfing_beach_for_kyogre_switch(
            active_id, switch_index, bench_attacker_index1, stadium_id
        ):
            return 750
        return 100
    if card_id in (Team_Rockets_Petrel, Team_Rockets_Receiver):
        return 150
    return 0


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

    # Hand quality checks: this deck has many Water Energy cards, so we
    # proactively seek draw/search supporters when the hand is clunky.
    non_energy_hand_count = my_state.handCount - hand_counts[Basic_Water_Energy]
    need_trolley = hand_counts[Precious_Trolley] == 0 and field_counts[Snover] + field_counts[Mega_Abomasnow_ex] <= 1
    need_mega = field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex] == 0
    need_kyogre = field_counts[Kyogre] + hand_counts[Kyogre] == 0
    hand_stuck = my_state.handCount <= 3 or non_energy_hand_count <= 2 or hand_counts[Basic_Water_Energy] >= 4
    need_draw_support = hand_stuck or need_trolley or need_mega or need_kyogre

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

    stadium_id = 0
    for card in state.stadium:
        if card != None:
            stadium_id = card.id
    
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
                elif context == SelectContext.TO_BENCH:
                    # When choosing a Pokémon to Bench.
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
                elif context == SelectContext.TO_HAND:
                    supporter_score = supporter_search_score(
                        card.id,
                        my_state,
                        field_counts,
                        hand_counts,
                        active_id,
                        switch_index,
                        bench_attacker_index1,
                        stadium_id,
                    )
                    if supporter_score != None:
                        score = supporter_score
                    elif card.id == Snover:
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
                    # Receiver replays a supporter from the discard pile.
                    if o.area == AreaType.DISCARD and o.playerIndex == my_index:
                        supporter_score = supporter_search_score(
                            card.id,
                            my_state,
                            field_counts,
                            hand_counts,
                            active_id,
                            switch_index,
                            bench_attacker_index1,
                            stadium_id,
                        )
                        if supporter_score != None:
                            score = supporter_score
                    # When choosing cards to discard from hand.
                    elif card.id == Basic_Water_Energy:
                        score += 100  # Prioritize Basic Water Energy for discard.
                    elif card.id == Mega_Abomasnow_ex:
                        score += 10
                    elif card.id == Carmine:
                        if hand_counts[Lillie_Determination] >= 1:
                            # If Lillie Determination is in the hand, Carmine is unnecessary.
                            score += 30
                    elif card.id in (Team_Rockets_Petrel, Team_Rockets_Receiver):
                        if hand_counts[Lillie_Determination] >= 1:
                            score += 30
                        if card.id == Team_Rockets_Petrel and hand_counts[Team_Rockets_Receiver] >= 1:
                            score += 80  # Receiver can reuse Petrel from discard.
                    elif card.id == Lillie_Determination:
                        score -= 20

                    if hand_counts[card.id] >= 2:
                        score += 500  # Prioritize discarding duplicate cards.
                    hand_counts[card.id] -= 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            score = 10000
            if card.id == Ultra_Ball:
                if hand_counts[Basic_Water_Energy] >= 3 or (my_state.handCount >= 4 and (field_counts[Mega_Abomasnow_ex] + hand_counts[Mega_Abomasnow_ex] == 0 or field_counts[Mega_Abomasnow_ex] + field_counts[Snover] == 0 or field_counts[Kyogre] == 0)):
                    # Only use if Water Energy is abundant or key cards are missing.
                    score = 4000
                else:
                    score = -1
            elif card.id == Carmine:
                if field_counts[Snover] >= 1 and hand_counts[Mega_Abomasnow_ex] >= 1:
                    score = -1
                else:
                    score = 5400 if need_draw_support else 3000
            elif card.id == Lillie_Determination:
                if field_counts[Snover] >= 1 and field_counts[Mega_Abomasnow_ex] == 0 and hand_counts[Mega_Abomasnow_ex] >= 1:
                    score = -1
                else:
                    score = 5500 if need_draw_support else 3100  # Prioritize over Carmine.
            elif card.id == Team_Rockets_Petrel:
                if field_counts[Snover] >= 1 and field_counts[Mega_Abomasnow_ex] == 0 and hand_counts[Mega_Abomasnow_ex] >= 1:
                    score = -1
                else:
                    # Treat as extra copies of draw supporters when hand quality is poor.
                    score = 5600 if need_draw_support else 3050
            elif card.id == Team_Rockets_Receiver:
                discard_support = (
                    discard_counts[Carmine]
                    + discard_counts[Lillie_Determination]
                    + discard_counts[Team_Rockets_Petrel]
                )
                if field_counts[Snover] >= 1 and field_counts[Mega_Abomasnow_ex] == 0 and hand_counts[Mega_Abomasnow_ex] >= 1:
                    score = -1
                elif discard_support >= 1:
                    score = 6000 if need_draw_support else 3200  # Replay draw supporters.
                elif field_counts[Mega_Abomasnow_ex] + field_counts[Snover] == 0:
                    score = 4300 if need_draw_support else 3000  # Early game: build the discard pile.
                else:
                    score = 3500 if need_draw_support else -1
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
