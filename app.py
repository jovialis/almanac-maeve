import os

from flask import Flask
from flask import request

from searcher import Searcher
import indexer as indexer
from exporter import export_node_contents
from paginator import Paginator

app = Flask(__name__)

searcher = Searcher()
searcher.train()


@app.route('/do/index-all', methods=['POST'])
def index_all_nodes():
    # extract node id and index
    indexer.index_all_nodes()

    return {"success": "true"}


@app.route('/do/index', methods=['POST'])
def index_node():
    # extract node id and index
    node_id = request.args['node']
    indexer.index_by_id(node_id)

    return {"success": "true"}


@app.route('/search', methods=['GET'])
def do_search():
    # extract query string and query settings
    query = request.args['query']

    # pull out paginate option
    paginate = bool(request.args.get("paginate", True))

    # pull out current page option
    page = int(request.args.get("page", 1))

    # search for nodes
    nodes = searcher.search(query)
    exported_nodes = list(map(lambda n: export_node_contents(n), nodes))

    res = {
        "query": query
    }

    if paginate:
        paginator = Paginator(exported_nodes, page)
        res["results"] = paginator.export_page_items()
        res["pagination"] = {
            "hits": paginator.hits(),
            "limit": paginator.page_limit(),
            "page": paginator.cur_page(),
            "totalPages": paginator.num_pages(),
            "hasNextPage": paginator.has_next_page(),
            "nextPage": paginator.next_page(),
            "hasPrevPage": paginator.has_prev_page(),
            "prevPage": paginator.prev_page()
        }
    else:
        res["results"] = exported_nodes

    # return json result
    return res


if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', "5000")))
