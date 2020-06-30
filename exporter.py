

# exports a node
# generic parent class
class NodeExporter:

    def __init__(self, node):
        self.node = node

    def export(self):
        export = {
            "name": self.node["name"],
            "model": self.node["primaryModelType"],
            "link": self.__node_link()
        }

        return export

    def __node_link(self):
        return "/nodes/" + self.node["primaryModelType"].lower() + "/" + self.node["nodeId"]


class CourseNodeExporter(NodeExporter):

    def export(self):
        # grab the default scraped info
        export = super().export()

        # 		subjects: this.subjects.map(s => s.name),
        # 		abbreviations: this.abbreviations.map(a => a.name),
        # 		description: this.description
        export["overview"] = {
            "subjects": list(map(lambda x: x["name"], self.node["subjects"])),
            "abbreviations": list(map(lambda x: x["name"], self.node["abbreviations"])),
            "description": self.node["description"]
        }

        return export


# supported scraper types
__node_special_exporters_map = {
    'CourseNode': CourseNodeExporter
}


def export_node_contents(node):
    node_type = node["nodeType"]

    # fetch an exporter, defaulting to NodeExporter
    exporter_type = __node_special_exporters_map.get(node_type, NodeExporter)

    # export
    exporter = exporter_type(node)
    exported_contents = exporter.export()

    return exported_contents
