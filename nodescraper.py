_NAME_WEIGHT = 2
_DEFAULT_WEIGHT = 1


# generic parent class
class NodeScraper:

    def __init__(self, node):
        self.node = node

    def extract_content(self):
        # by default, just extract the name
        name = (self.node["name"], _NAME_WEIGHT)
        return [name]


class CourseNodeScraper(NodeScraper):

    def extract_content(self):
        # grab the default scraped info
        content = super().extract_content()

        # extract description and subjects
        description = (self.node["description"] or "", _DEFAULT_WEIGHT)
        subjects = map(lambda x: (x["name"], _DEFAULT_WEIGHT), self.node["subjects"])

        content.extend(subjects)
        content.append(description)

        return content


class SubjectNodeScraper(NodeScraper):

    def extract_content(self):
        # grab the default
        content = super().extract_content()

        # throw in abbreviation
        abbreviation = (self.node["abbreviation"], _DEFAULT_WEIGHT)

        content.append(abbreviation)

        return content


# supported scraper types
__node_scraper_map = {
    'CourseNode': CourseNodeScraper,
    'SubjectNode': SubjectNodeScraper,
    'InstructorNode': NodeScraper
}


def scrape_node_contents(node):
    node_type = node["nodeType"]

    # ignore if node type not supported
    if node_type not in __node_scraper_map.keys():
        return

    # instantiate a scraper
    scraper_type = __node_scraper_map[node_type]
    scraper = scraper_type(node)

    # grab content from scraper
    content = scraper.extract_content()
    return content
