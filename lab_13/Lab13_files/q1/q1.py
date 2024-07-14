import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


''' Do not change anything in this function '''
def generate_random_profiles(num_voters, num_candidates):
    '''
        Generates a NumPy array where row i denotes the strict preference order of voter i
        The first value in row i denotes the candidate with the highest preference
        Result is a NumPy array of size (num_voters x num_candidates)
    '''
    return np.array([np.random.permutation(np.arange(1, num_candidates+1)) 
            for _ in range(num_voters)])


def find_winner(profiles, voting_rule):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        In STV, if there is a tie amongst the candidates with minimum plurality score in a round, then eliminate the candidate with the lower index
        For Copeland rule, ties among pairwise competitions lead to half a point for both candidates in their Copeland score

        Return: Index of winning candidate (1-indexed) found using the given voting rule
        If there is a tie amongst the winners, then return the winner with a lower index
    '''

    winner_index = 0
    
    # TODO

    if voting_rule == 'plurality':
        n = len(profiles)
        m = len(profiles[0])

        votes = np.zeros(m)

        for i in range(n):
            votes[profiles[i][0]-1] += 1

        winner_index = np.argmax(votes) + 1

    elif voting_rule == 'borda':
        n = len(profiles)
        m = len(profiles[0])

        votes = np.zeros(m)

        for i in range(n):
            for j in range(m):
                votes[profiles[i][j]-1] += (m-(j+1))

        winner_index = np.argmax(votes) + 1

    elif voting_rule == 'stv':
        n = len(profiles)
        m = len(profiles[0])

        removed_candidates = []

        for _ in range(m-1):
            votes = np.zeros(m)
            for i in range(n):
                # k = 0
                # if profiles[i][0] not in removed_candidates:
                #     votes[profiles[i][0]] += 1
                # else:
                #     votes[profiles[i][0]] = n+1
                # while (profiles[i][k]-1) in removed_candidates:
                #     k += 1
                #     print("hi")
                votes[profiles[i][0]-1] += 1
            for t in removed_candidates:
                votes[t] = n+1
            worst_candidate = np.argmin(votes)
            removed_candidates.append(worst_candidate)

            for i in range(n):
                worst_candidate_posn = np.where(profiles[i] == (worst_candidate+1))[0]
                # print(worst_candidate_posn)
                # print(m)
                if len(worst_candidate_posn) == 1:
                    k = len(removed_candidates)
                    for j in range(worst_candidate_posn[0], m-k-1):
                        profiles[i][j], profiles[i][j+1] = profiles[i][j+1], profiles[i][j]

        for j in range(m):
            if j not in removed_candidates:
                winner_index = j + 1
                break
            

    elif voting_rule == 'copeland':
        n = len(profiles)
        m = len(profiles[0])

        copeland_scores = np.zeros(m)

        profiles_posn = np.zeros_like(profiles)

        for i in range(n):
            for j in range(m):
                profiles_posn[i][j] = np.where(profiles[i] == (j+1))[0]

        for a in range(m):
            for b in range(a+1,m):
                x = 0
                y = 0
                for i in range(n):
                    if profiles_posn[i][a] < profiles_posn[i][b]:
                        x += 1
                    else:
                        y += 1

                if x > y:
                    copeland_scores[a] += 1
                elif x < y:
                    copeland_scores[b] += 1
                elif x == y:
                    copeland_scores[a] += 0.5
                    copeland_scores[b] += 0.5

        winner_index = np.argmax(copeland_scores) + 1

    # END TODO

    return winner_index


def find_winner_average_rank(profiles, winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        winner is the index of the winning candidate for some voting rule (1-indexed)

        Return: The average rank of the winning candidate (rank wrt a voter can be from 1 to num_candidates)
    '''

    average_rank = 0

    # TODO
    n = len(profiles)
    m = len(profiles[0])

    for i in range(n):
        rank = np.where(np.array(profiles[i]) == winner)[0] + 1
        average_rank += rank

    average_rank = average_rank/n

    # END TODO

    return average_rank


def check_manipulable(profiles, voting_rule, find_winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        find_winner is a function that takes profiles and voting_rule as input, and gives the winner index as the output
        It is guaranteed that there will be at most 8 candidates if checking manipulability of a voting rule

        Return: Boolean representing whether the voting rule is manipulable for the given preference profiles
    '''

    manipulable = False

    # TODO

    n = len(profiles)
    m = len(profiles[0])

    initial_winner = find_winner(profiles, voting_rule)

    for i in range(n):
        initial_profile = profiles[i]
        profiles_new = np.copy(profiles)

        for new_profile in itertools.permutations(initial_profile):
            profiles_new[i] = new_profile

            new_winner = find_winner(profiles_new, voting_rule)

            if new_winner != initial_winner:
                if np.where(initial_profile == new_winner)[0] < np.where(initial_profile == initial_winner)[0]:
                    manipulable = True
                    return manipulable
    
    # END TODO

    return manipulable


if __name__ == '__main__':
    np.random.seed(420)

    num_tests = 200
    voting_rules = ['plurality', 'borda', 'stv', 'copeland']

    average_ranks = [[] for _ in range(len(voting_rules))]
    manipulable = [[] for _ in range(len(voting_rules))]
    for _ in tqdm(range(num_tests)):
        # Check average ranks of winner
        num_voters = np.random.choice(np.arange(80, 150))
        num_candidates = np.random.choice(np.arange(10, 80))
        profiles = generate_random_profiles(num_voters, num_candidates)

        for idx, rule in enumerate(voting_rules):
            winner = find_winner(profiles, rule)
            avg_rank = find_winner_average_rank(profiles, winner)
            average_ranks[idx].append(avg_rank / num_candidates)

        # Check if profile is manipulable or not
        num_voters = np.random.choice(np.arange(10, 20))
        num_candidates = np.random.choice(np.arange(4, 8))
        profiles = generate_random_profiles(num_voters, num_candidates)
        
        for idx, rule in enumerate(voting_rules):
            manipulable[idx].append(check_manipulable(profiles, rule, find_winner))


    # Plot average ranks as a histogram
    for idx, rule in enumerate(voting_rules):
        plt.hist(average_ranks[idx], alpha=0.8, label=rule)

    plt.legend()
    plt.xlabel('Fractional average rank of winner')
    plt.ylabel('Frequency')
    plt.savefig('average_ranks.jpg')
    
    # Plot bar chart for fraction of manipulable profiles
    manipulable = np.sum(np.array(manipulable), axis=1)
    manipulable = np.divide(manipulable, num_tests)
    plt.clf()
    plt.bar(voting_rules, manipulable)
    plt.ylabel('Manipulability fraction')
    plt.savefig('manipulable.jpg')