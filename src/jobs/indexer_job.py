import datetime

from src.rag.engine import RAGEngine
from src.redmine.client import RedmineClient


def run_sync(last_run_date: str | None = None):
    engine = RAGEngine()
    redmine = RedmineClient()

    # Si no hay fecha, es la carga inicial (trae todo)
    # Si hay fecha, trae solo los modificados/creados desde entonces
    query_params = {}
    if last_run_date:
        query_params["updated_on"] = f">={last_run_date}"

    issues = redmine.list_issues(fetch_all=True, status_id="*", **query_params)

    if issues:
        print(f"Indexando {len(issues)} issues...")
        engine.index_redmine_issues(issues)
    else:
        print("No hay issues nuevos o modificados.")


if __name__ == "__main__":
    # Aquí puedes leer la fecha de la última ejecución desde un archivo o BD
    # Para el cron job, puedes calcular la fecha de hace 24 horas:
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    run_sync(last_run_date=yesterday)
