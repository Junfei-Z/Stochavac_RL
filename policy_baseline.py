# policy_baseline.py

def baseline_policy(state, G, budget=1):
    vaccinated = {}
    count = 0
    priority = ['high_risk', 'high_contact', 'baseline']
    for group in priority:
        for node in G.nodes:
            if G.nodes[node]['group'] == group and state[node] == 'S':
                vaccinated[node] = 1
                count += 1
                if count >= budget:
                    return vaccinated
    return vaccinated
