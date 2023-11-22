import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_matrix(matrix, approaches, indicator, Interpretation):
    os.makedirs('pic', exist_ok=True)
    fig, ax = plt.subplots()
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(matrix, cmap=plt.cm.jet)
    plt.colorbar(ax.imshow(matrix, cmap=plt.cm.jet))

    for (i, j), z in np.ndenumerate(matrix):
        if i==j:
            ax.text(j, i, 'NA', ha='center', va='center', color='red',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.title(f"matrix_{indicator}_DownTo_{Interpretation}")
    plt.xticks(range(matrix.shape[0]), approaches)
    plt.yticks(range(matrix.shape[0]), approaches)
    plt.tight_layout()
    plt.savefig(os.path.join("pic", f"matrix_{indicator}_DownTo_{Interpretation}.png"), format='png', dpi=300)
    plt.show()

if __name__ == "__main__":
    approaches = ['H_1', 'H_2', 'H_3', 'H_4', 'standard']
    indicators = ['reeval_IGD','budget']
    Interpretations = ['betterLarge', 'betterMedium', 'betterSmall', 'betterNegl']
    # Interpretations = ['betterMedium']
    # for indicator in indicators:
    #     matrix = np.zeros((len(approaches), len(approaches)))
    #     for Interpretation in Interpretations:
    #         for r_idx, approach_1 in enumerate(approaches):
    #             for c_idx, approach_2 in enumerate(approaches):
    #                 if approach_1 != approach_2:
    #                     COMP = pd.read_csv(f"A1_{approach_1}_A2_{approach_2}_{indicator}.csv")
    #                     matrix[r_idx, c_idx] +=  COMP[Interpretation].sum()
            
    #         plot_matrix(matrix, approaches, indicator, Interpretation)
    
    for indicator in indicators:
        matrix = np.zeros((len(approaches), len(approaches)))
        for r_idx, approach_1 in enumerate(approaches):
            for c_idx, approach_2 in enumerate(approaches):
                if approach_1 != approach_2:
                    COMP = pd.read_csv(f"A1_{approach_1}_A2_{approach_2}_{indicator}.csv")
                    matrix[r_idx, c_idx] +=  COMP['diff_cost'].sum()
        
        plot_matrix(matrix, approaches, indicator, 'diff_cost')