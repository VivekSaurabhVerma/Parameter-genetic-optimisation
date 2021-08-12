import random
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def create_population(n, params):
    """ Create initial population """
    population = []
    for i in range(n):
        individu = []
        for param in params:
            p = params[param]
            if p[0] == 'int':
                individu.append(random.randint(p[1], p[2]))
            elif p[0] == 'float':
                r = p[1] + random.random() * (p[2] - p[1])
                individu.append(r)
            elif p[0] == 'list':
                individu.append(random.choice(p[1]))
        population.append(individu)
    return population


def evaluate(params, individu, model, X, y, val_data, metric):
    """ Evaluate a genome """
    dic = {}
    for i, p in enumerate(params):
        dic[p] = individu[i]
    sklearn_model = model.set_params(**dic)

    if type(val_data) is float or type(val_data) is int:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_data)
    else:
        X_train, X_test, y_train, y_test = X, y, val_data[0], val_data[1]

    sklearn_model.fit(X_train, y_train)
    y_pred = sklearn_model.predict(X_test)

    s = metric(y_test, y_pred)
    return s


def mutate(individu, params):
    place_to_change = random.randrange(0, len(individu))
    param_to_change = [p for p in params][place_to_change]
    p = params[param_to_change]
    if p[0] == 'int':
        individu[place_to_change] = random.randint(p[1], p[2])
    elif p[0] == 'float':
        r = p[1] + random.random() * (p[2] - p[1])
        individu[place_to_change] = r
    elif p[0] == 'list':
        individu[place_to_change] = random.choice(p[1])
    return individu


def evaluate_population(population, params, model, X, y, val_data, metric):
    """ Grade the population. Return a list of tuple (individual, fitness) sorted from most graded to less graded. """
    graded_individual = []
    for individual in population:
        graded_individual.append((individual, evaluate(params, individual, model, X, y, val_data, metric)))

    return sorted(graded_individual, key=lambda x: x[1], reverse=True)


def evolve_population(population, params, model, metric, X, y, val_data=0.3, graded_prop=0.2, non_graded_prop=0.05,
                      mutation_rate=0.1):
    """ Make the given population evolving to his next generation. """

    # Evaluate population
    sorted_pop_fit = evaluate_population(population, params, model, X, y, val_data, metric)
    n = len(sorted_pop_fit)

    avg_fitness = sum([i[1] for i in sorted_pop_fit]) / n

    # Select parents
    sorted_pop = [i[0] for i in sorted_pop_fit]
    parents = sorted_pop[:int(graded_prop * n)]
    best_ind = sorted_pop[0]

    # Randomly add other individuals to promote genetic diversity
    for individual in sorted_pop[int(graded_prop * n):]:
        if random.random() < non_graded_prop:
            parents.append(individual)

    # Mutate some individuals
    for i in range(len(parents)):
        if random.random() < mutation_rate:
            parents[i] = mutate(parents[i], params)

            # Crossover parents to create children
    parents_len = len(parents)
    desired_len = n - parents_len
    children = []
    while len(children) < desired_len:
        father = random.choice(parents)
        mother = random.choice(parents)
        if True:  # father != mother:
            child = []
            mask = [random.random() > 0.5 for _ in range(len(father))]
            for i in range(len(mask)):
                if mask[i]:
                    child.append(father[i])
                else:
                    child.append(mother[i])

            children.append(child)

    # The next generation is ready
    parents.extend(children)

    return parents, avg_fitness, best_ind


def geneticSearchCV(params, model, metric, X, y, val_data=0.2,
                    population_size=100, n_generations=10,
                    graded_prop=0.2, non_graded_prop=0.05, mutation_rate=0.1,
                    verbose=True, checkpoint=True):
    """ Optimizes hyperparameters with a genetic algorithm """

    population = create_population(population_size, params)

    for generation in range(n_generations):
        population, avg_fitness, best_ind = evolve_population(population, params, model, metric, X, y,
                                                              graded_prop=graded_prop,
                                                              non_graded_prop=non_graded_prop,
                                                              mutation_rate=mutation_rate)

        if verbose:
            print(f'Population {generation + 1} : Avg fitness = {avg_fitness}')
        if checkpoint:
            dic = {}
            for i, p in enumerate(params):
                dic[p] = best_ind[i]
            f = open('checkpoint_best_genome.txt', 'w')
            f.write(str(dic))
            f.close()

    if verbose:
        print('\nEvaluation finale')
    score = evaluate_population(population, params, model, X, y, val_data, metric)
    if verbose:
        print('Best score : ', score[0][1])
        print('Best genome : ', score[0][0])

    dic = {}
    for i, p in enumerate(params):
        dic[p] = score[0][0][i]
    model.set_params(**dic)

    return dic, model


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    model = make_pipeline(StandardScaler(), RandomForestClassifier())

    params = {
        'randomforestclassifier__n_estimators': ['int', 1, 1000],
        'randomforestclassifier__max_depth': ['int', 1, 1000],
        'randomforestclassifier__criterion': ['list', ['gini', 'entropy']],
        'randomforestclassifier__min_samples_split': ['float', 0, 1 - 1e-9],
    }
    params, best_model = geneticSearchCV(params, model, accuracy_score, X, y, val_data=0.2,
                                         population_size=60, n_generations=100,
                                         graded_prop=0.2, non_graded_prop=0.05, mutation_rate=0.1,
                                         verbose=True, checkpoint=True)
