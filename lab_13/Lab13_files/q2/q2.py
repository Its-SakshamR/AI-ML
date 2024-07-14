import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import copy 
def Gale_Shapley(suitor_prefs, reviewer_prefs) -> Dict[str, str]:
    '''
        Gale-Shapley Algorithm for Stable Matching

        Parameters:

        suitor_prefs: dict - Dictionary of suitor preferences
        reviewer_prefs: dict - Dictionary of reviewer preferences

        Returns:

        matching: dict - Dictionary of suitor matching with reviewer
    '''
    # suitor_prefs_copy = copy.deepcopy(suitor_prefs)
    suitor_prefs_copy={}
    for i in suitor_prefs:
        suitor_prefs_copy[i]=suitor_prefs[i].copy()
    matching = {}
    ## TODO: Implement the Gale-Shapley Algorithm
    # free_suitors = []
    free_suitors = list(suitor_prefs) # All suitors are free initially
    isReviewerFree = {}
    reviewers = list(reviewer_prefs)
    for i in reviewers:
        isReviewerFree[i] = None # Initially all reviewers are free 

    while free_suitors:
        free_suitor = free_suitors.pop(0)
        highest_reviewer = suitor_prefs_copy[free_suitor].pop(0) # Get the most preferred reviewer
        # if(len(suitor_prefs['F'])==1):
        #     print(isReviewerFree[highest_reviewer] is None)
        if (isReviewerFree[highest_reviewer]):
            previous_suitor = isReviewerFree[highest_reviewer]
            if reviewer_prefs[highest_reviewer].index(previous_suitor) < reviewer_prefs[highest_reviewer].index(free_suitor) :
                # She loves prev suitor more
                # free suitor is added back to the list
                free_suitors.append(free_suitor)
            else : # She loves free_suitor more
                matching[free_suitor] = highest_reviewer # Free suitor engaged
                previous_suitor = isReviewerFree[highest_reviewer] 
                matching[previous_suitor] = None # Prev suitor becomes free

                # if highest_reviewer in suitor_prefs[previous_suitor] :
                # suitor_prefs[previous_suitor].remove(highest_reviewer) # He can't propose to 
                free_suitors.append(previous_suitor) # Add him to free list
                isReviewerFree[highest_reviewer] = free_suitor # Update the choice of highest reviewer
        else : # She is engaged to somebody. How will u find that 
            matching[free_suitor] = highest_reviewer
            isReviewerFree[highest_reviewer] = free_suitor # Forgot

    ## END TODO
    # print(suitor_prefs['A'])
    # for i in matching:
    #     suitor_prefs[i].insert(0,matching[i])
    return matching

def avg_suitor_ranking(suitor_prefs: Dict[str, List[str]], matching: Dict[str, str]) -> float:
    '''
        Calculate the average ranking of suitor in the matching
        
        Parameters:
        
        suitor_prefs: dict - Dictionary of suitor preferences
        matching: dict - Dictionary of matching
        
        Returns:
        
        avg_suitor_ranking: float - Average ranking of suitor
    '''
    # print(suitor_prefs['A'])
    avg_suitor_ranking = 0

    ## TODO: Implement the average suitor ranking calculation

    for suitor in matching :
        avg_suitor_ranking += suitor_prefs[suitor].index(matching[suitor])
    
    avg_suitor_ranking = avg_suitor_ranking / len(suitor_prefs)
    
    avg_suitor_ranking += 1 # 1 based indexing

    ## END TODO

    assert type(avg_suitor_ranking) == float

    return avg_suitor_ranking

def avg_reviewer_ranking(reviewer_prefs: Dict[str, List[str]], matching: Dict[str, str]) -> float:
    '''
        Calculate the average ranking of reviewer in the matching
        
        Parameters:
        
        reviewer_prefs: dict - Dictionary of reviewer preferences
        matching: dict - Dictionary of matching
        
        Returns:
        
        avg_reviewer_ranking: float - Average ranking of reviewer
    '''

    avg_reviewer_ranking = 0

    ## TODO: Implement the average reviewer ranking calculation
    for suitor in matching :
        reviewer = matching[suitor]
        avg_reviewer_ranking += reviewer_prefs[reviewer].index(suitor)
    
    avg_reviewer_ranking = avg_reviewer_ranking / len(matching)
    avg_reviewer_ranking += 1

    ## END TODO

    assert type(avg_reviewer_ranking) == float

    return avg_reviewer_ranking

def get_preferences(file) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    '''
        Get the preferences from the file
        
        Parameters:
        
        file: file - File containing the preferences
        
        Returns:
        
        suitor_prefs: dict - Dictionary of suitor preferences
        reviewer_prefs: dict - Dictionary of reviewer preferences
    '''
    suitor_prefs = {}
    reviewer_prefs = {}

    for line in file:
        if line[0].islower():
            reviewer, prefs = line.strip().split(' : ')
            reviewer_prefs[reviewer] = prefs.split()

        else:
            suitor, prefs = line.strip().split(' : ')
            suitor_prefs[suitor] = prefs.split()
        
    return suitor_prefs, reviewer_prefs


if __name__ == '__main__':

    avg_suitor_ranking_list = []
    avg_reviewer_ranking_list = []

    for i in range(100):
        with open('data/data_'+str(i)+'.txt', 'r') as f:
            suitor_prefs, reviewer_prefs = get_preferences(f)

            # suitor_prefs = {
            #     'A': ['a', 'b', 'c'],
            #     'B': ['c', 'b', 'a'],
            #     'C': ['c', 'a', 'b']
            # }

            # reviewer_prefs = {
            #     'a': ['A', 'C', 'B'],
            #     'b': ['B', 'A', 'C'],
            #     'c': ['B', 'A', 'C']
            # }

            matching = Gale_Shapley(suitor_prefs, reviewer_prefs)

            avg_suitor_ranking_list.append(avg_suitor_ranking(suitor_prefs, matching))
            avg_reviewer_ranking_list.append(avg_reviewer_ranking(reviewer_prefs, matching))

    plt.hist(avg_suitor_ranking_list, bins=10, label='Suitor', alpha=0.5, color='r')
    plt.hist(avg_reviewer_ranking_list, bins=10, label='Reviewer', alpha=0.5, color='g')

    plt.xlabel('Average Ranking')
    plt.ylabel('Frequency')

    plt.legend()
    plt.savefig('q2.png')


    