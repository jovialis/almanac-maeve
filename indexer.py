from pymongo import MongoClient
from bson.objectid import ObjectId
import datetime
import math

from nodescraper import scrape_node_contents
from analyzecontent import AnalyzeContent

__client = MongoClient()
__models_db = __client['modellingDB']
__search_db = __client['searchDB']

__nodes_collection = __models_db['nodes']
__indexes_collection = __search_db['docs']
__lemmas_collection = __search_db['lemmas']


def index_all_nodes():
    start = datetime.datetime.now()
    nodes = list(__nodes_collection.find({"$or": [
        {"nodeType": "CourseNode"},
        {"nodeType": "SubjectNode"},
        {"nodeType": "InstructorNode"}
    ]}))

    print('Preparing to index ' + str(len(nodes)) + ' nodes')
    percent_marker = math.floor(len(nodes) / 100)

    for i, node in enumerate(nodes):
        index(node, output=False)

        if i % percent_marker == 0:
            print(str(i / percent_marker) + '% Done')

    print('Done in ' + str((datetime.datetime.now()) - start))


def index_by_id(node_id_string):
    node_id = ObjectId(node_id_string)

    # discover the node by its id and index
    node = __nodes_collection.find_one({"_id": node_id})
    if node is not None:
        index(node)


def index(node, output=True):
    node_id = node['_id']

    # extract text content from node
    text_content = scrape_node_contents(node)
    analyzer = AnalyzeContent(text_weights=text_content)

    node_lemmas = analyzer.content_token_list()

    # create contents
    index_contents = {
        "node": node_id,
        "lemmas": node_lemmas,
        "lemmaWeights": analyzer.content_token_weight_map(),
        "body": analyzer.content_tokens_flattened_raw(),
        "bodyLength": len(analyzer.content_tokens_flattened_raw())
    }

    # update the document
    res = __indexes_collection.update_one(
        {
            "node": node_id
        }, {
            "$set": index_contents,
        },
        upsert=True
    )

    # insert document into the doc list for each of the lemmas
    for lemma in node_lemmas:
        # insert node id to lemma doc
        __lemmas_collection.update_one(
            {
                "lemma": lemma
            }, {
               "$addToSet": {
                   "docs": res.upserted_id
               }
            },
            upsert=True
        )

    # remove document from any lemma docs that are no longer relevant
    __lemmas_collection.update(
        {
            "docs": res.upserted_id,
            "lemma": {
                "$nin": node_lemmas
            }
        },
        {
            "$pull": {
                "docs": res.upserted_id
            }
        }
    )

    if output:
        print('Indexed node named: ' + node["name"])
