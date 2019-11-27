set -euxo pipefail

binpath=$(dirname "$0")

toy(){
    pipenv run python $binpath/test_kmm.py -d $binpath/happyface.data -g $binpath/happyface.ground 
}
all(){
    pipenv run python $binpath/batchRun.py
    pipenv run python $binpath/batchPlot.py
}

usage(){
    echo "./<path to KMM-Python>/bin/test.sh toy | all"
}
if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
else
    case $1 in
        toy)                  toy
                                ;;
        all)                  all
                                ;;
        launch)                	launch $2
                                ;;
        destroy)				destroy $2
        						;;
        update)					updata_alluxio $2
        						;;
		conf_alluxio)			configure_alluxio $2
								;;             
		man_start)				manual_restart $2
								;;
        * )                     usage
    esac
fi