from task_runner import TaskRunner
import plotter_classification
import clean_up

if __name__ == '__main__':
    clean_up.do_it()
    tag = "p24"
    tasks = {
        "algorithms" : ["c4"],
        "datasets": ["ghisaconus_health"],
        "target_sizes" : list(range(30,1))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True, test=False)
    summary, details = ev.evaluate()
    plotter_classification.plot_combined(sources=["p22","p23","p24"],only_algorithms=["c2","c3","c4"], only_datasets=["ghisaconus_health"])