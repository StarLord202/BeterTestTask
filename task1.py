import numpy
import abc
import time
import numpy as np


class BasicDistrCalculator(abc.ABC):

    """Базовий клас який реалізує інтерфейс обчислювача імовірностей"""

    @abc.abstractmethod
    def get_distr(self, p1, p2, s1, s2) -> dict:
        """базовий метод який має бути реалізований у всіх нащадків цього класу"""
        pass

    def _validate_inputs(self, p1, p2, s1, s2):

        if not isinstance(s1, int) or not isinstance(s2, int):
            return False

        if s1 < 0 or s2 < 0 or s1 + s2 > 20:
            return False


        if not isinstance(p1, (int, float)) or not isinstance(p2, (int, float)):
            return False

        if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1:
            return False

        return True

    def _get_pl(self, n) -> int:

        """метод який приймає номер раунду і видає номер гравця який зараз подає 0 якщо перший гравець і 1 якщо другий гравець"""

        if n < 2:
            return 0

        else:
            if n % 2 == 0:
                if n % 4 == 0:
                    return 0
                else:
                    return 1
            else:
                return self._get_pl(n - 1)



class GreedyTree(BasicDistrCalculator):

    """Клас який реалізує метод жадібного дерева без оптимізації експоненційно зростаючих розрахунків"""

    def get_distr(self, p1, p2, s1, s2):

        if not self._validate_inputs(p1, p2, s1, s2):
            raise ValueError("Wrong arguments")


        """
        Ідея цього алгоритму полягає в наступному:
        ми стартуємо з початкового положення (s1, s2) це є перша нода нашого дерева
        далі ми визначаємо імовірності піти вліво(перший гравець забив) або вправо(другий гравець забив)
        додаємо ці ноди на наступни рівень і на наступній ітерації вважаємо цей рівень початковим і повторюємо процесс
        для кожної ноди з цього рівня і перевіряємо чи ми не прийшли в один з фінальних станів,
        якщо так то ця нода вже не розглядається, повторюємо це до тих пір, доки наступний рівень не буде пустим

        тут кожна нода визначається рахунком першого гравця і імовірністю того, що перший гравець заб'є
        для зручності цю інформацію зберігаємо у різних списках
        """

        left_probs = (p1, p2)
        cur_glob_count = s1 + s2
        current_left_counts = [s1] #ініціалізуємо початковий стан
        current_probas = [1]
        distributions = {} #тут міститимуться фінальні рахунки та їх імовірності


        def add_to_distributions(x, proba):
            """ця функція перевіряє чи рахунок вже є у списку, якщо є то сумуємо імовірності, якщо немає то створюємо запис"""
            if x in distributions:
                distributions[x] += proba
            else:
                distributions[x] = proba


        while True:
            "визначаємо імовірність піти вліво для цього рівня"
            cur_plr = self._get_pl(cur_glob_count)
            cur_lt_pb = left_probs[cur_plr]#для цього рівня імовірність піти вліво така
            next_counts = []
            next_probas = []
            for i in range(len(current_probas)):
                "перебираємо ноди тобто рахунок першого гравця і імовріність прийти в цю ноду"
                cur_lt_count = current_left_counts[i]
                cur_proba = current_probas[i]

                "перевіряємо чи нода відповідає фінальному стану"
                if cur_lt_count == 11:
                    add_to_distributions((11, cur_glob_count - 11), cur_proba)

                elif cur_lt_count + 11 == cur_glob_count:
                    add_to_distributions((cur_lt_count, 11), cur_proba)

                elif cur_glob_count == 20:
                    add_to_distributions((10, 10), cur_proba)

                else:
                    "якщо не відповідає то визначаємо наступні ноди і імовірності піти туди"
                    next_lt_count = cur_lt_count + 1
                    next_lt_proba = cur_proba*cur_lt_pb
                    next_rt_proba = cur_proba*(1-cur_lt_pb)

                    next_counts.extend([next_lt_count, cur_lt_count])
                    next_probas.extend([next_lt_proba, next_rt_proba])
                    """додаємо ці ноди та імовірності на наступний рівень
                       тобто в наступні рахунки записуємо можливі рахунки першого гравця
                       на 1 більше або такий самий як зараз і те саме робимо з імовірностями
                       """


            if len(next_counts) == 0:
                "перевіряємо чи наступний рівень не пустий"
                break

            else:
                "якщо ні то настпуний рівень стає початковим рівнем і процесс продовжується"
                current_probas = next_probas
                current_left_counts = next_counts
                cur_glob_count += 1



        return distributions






class OptimizedTree(BasicDistrCalculator):


    def get_distr(self, p1, p2, s1, s2):

        if not self._validate_inputs(p1, p2, s1, s2):
            raise ValueError("Wrong arguments")

        """Суть алгоритму та сама, тільки ми склеюємо між собою однакові ноди
           тобто сумуємо імовірності прийти в цю ноду і записуємо її один раз"""

        left_probs = (p1, p2)
        cur_glob_count = s1 + s2
        middle_distributions = {s1:1}
        final_distributions = {}


        def add_to_distributions(x, proba, distr):
            if x in distr:
                distr[x] += proba
            else:
                distr[x] = proba


        while True:
            cur_plr = self._get_pl(cur_glob_count)
            cur_lt_proba = left_probs[cur_plr]
            next_middle_distrs = {}
            for i in range(len(middle_distributions)):
                cur_distr = list(middle_distributions.items())[i]
                cur_lt_count, cur_proba = cur_distr

                if cur_lt_count == 11:
                    add_to_distributions((11, cur_glob_count - 11), cur_proba, final_distributions)

                elif cur_lt_count + 11 == cur_glob_count:
                    add_to_distributions((cur_lt_count, 11), cur_proba, final_distributions)

                elif cur_glob_count == 20:
                    add_to_distributions((10, 10), cur_proba, final_distributions)

                else:
                    next_lt_count = cur_lt_count + 1
                    next_lt_proba = cur_proba*cur_lt_proba
                    next_rt_proba = cur_proba*(1-cur_lt_proba)


                    """В цьому моменті ми дивимось, чи вже така нода існує, якщо так то сумуємо імовірності"""
                    add_to_distributions(next_lt_count, next_lt_proba, next_middle_distrs)
                    add_to_distributions(cur_lt_count, next_rt_proba, next_middle_distrs)

            if len(next_middle_distrs) == 0:
                break

            else:
                middle_distributions = next_middle_distrs
                cur_glob_count += 1

        return final_distributions




class MarkovCalc(BasicDistrCalculator):

    def __init_matrix(self, p1, p2, s1, s2):

        """Функція яка ініціалізує матрицю станів, станами тут є усі проміжні рахунки
           """

        states = []
        probs = (p1, p2)
        final_states = []

        """тут для зручності просто ініціалізуємо усі можливі рахунки"""

        for i in range(s1, 11):
            for j in range(s2, 11):
                states.append((i, j))

        final_states.append((10, 10))

        for k in range(s1, 10):
            states.append((k, 11))
            final_states.append((k, 11))

        for k in range(s2, 10):
            states.append((11, k))
            final_states.append((11, k))

        K = len(states)


        """словник зі станами та відповідними їм індексами"""
        states = {states[k]:k for k in range(K)}


        matrix = numpy.zeros((K, K))
        sums = list(states.keys())

        for k in range(K):

            """Проходимось по всім станам, якщо стан фінальний то нічого не робимо, якщо не фінальний то 
               додаємо до матриці імовірності перейти в два наступних стани"""

            cur_ind = states[sums[k]]
            cur_s1, cur_s2 = sums[k]

            if (cur_s1 == 11 or cur_s2 == 11) or ((cur_s1 == 10) and (cur_s2 == 10)):
                continue


            """тут як і в деревах визначаємо імовірність переходу"""
            cur_sum = cur_s1 + cur_s2
            cur_plr = self._get_pl(cur_sum)
            cur_prob = probs[cur_plr]


            """тут за станами визначаємо індекси відповідних їм колонок"""
            next_ind_1 = states[(cur_s1 + 1, cur_s2)]
            next_ind_2 = states[(cur_s1, cur_s2 + 1)]

            matrix[cur_ind, next_ind_1] = cur_prob
            matrix[cur_ind, next_ind_2] = 1 - cur_prob


        final_states_indexes = {state:states[state] for state in final_states}

        return matrix, final_states_indexes












    def get_distr(self, p1, p2, s1, s2):

        if not self._validate_inputs(p1, p2, s1, s2):
            raise ValueError("Wrong arguments")

        """отримуємо матрицю переходів для заданої конфігурації"""

        matrix, final_states_indexes = self.__init_matrix(p1, p2, s1, s2)

        cur_sum = s1 + s2

        """визначаємо скільки ще треба зробити кроків від цього стану щоби рахувати імовірності фінальних станів"""

        n = max(1, 11 - cur_sum)
        m = 20 - cur_sum

        np_indexes = np.asarray((list(final_states_indexes.values())))
        matrix_accumulator = numpy.linalg.matrix_power(matrix, n)
        probs_accumulator = matrix_accumulator[0, np_indexes]

        """рахуємо імовірності"""

        for _ in range(n, m):
            matrix_accumulator = matrix_accumulator @ matrix
            probs_accumulator += matrix_accumulator[0, np_indexes]


        result = {list(final_states_indexes.keys())[k]:probs_accumulator[k] for k in range(len(np_indexes))}

        return result


def timeit(func, n):
    def inner(*args, **kwargs):
        sum = 0
        for _ in range(n):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            sum += end - start
        return sum/n
    return inner

def main():
    s1 = 9
    s2 = 9
    p1 = 0.5
    p2 = 0.5

    greedy_func = GreedyTree().get_distr
    optim_func = OptimizedTree().get_distr
    markov_func = MarkovCalc().get_distr

    greedy_res = greedy_func(p1, p2, s1, s2)
    optim_res = optim_func(p1, p2, s1, s2)
    markov_res = markov_func(p1, p2, s1, s2)

    for key in greedy_res.keys():
        print(f"{key}: greedy {greedy_res[key]}, optim {optim_res[key]}, markov {markov_res[key]}")

    greedy_time = timeit(greedy_func, 100)(p1, p2, s1, s2)
    optim_time = timeit(optim_func, 100)(p1, p2, s1, s2)
    markov_time = timeit(markov_func, 100)(p1, p2, s1, s2)

    print(f"greedy time: {greedy_time}, optim time: {optim_time}, markov time: {markov_time}")

if __name__ == '__main__':
    main()

