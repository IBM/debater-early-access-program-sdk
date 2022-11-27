import logging

import docx
from docx.shared import Inches
from docx import Document

from debater_python_api.api.clients.key_point_analysis.utils import read_dicts_from_csv, create_dict_to_list, trunc_float


def add_bookmark(paragraph, bookmark_text, bookmark_name):
    run = paragraph.add_run()
    tag = run._r
    start = docx.oxml.shared.OxmlElement('w:bookmarkStart')
    start.set(docx.oxml.ns.qn('w:id'), '0')
    start.set(docx.oxml.ns.qn('w:name'), bookmark_name)
    tag.append(start)

    text = docx.oxml.OxmlElement('w:r')
    text.text = bookmark_text
    tag.append(text)

    end = docx.oxml.shared.OxmlElement('w:bookmarkEnd')
    end.set(docx.oxml.ns.qn('w:id'), '0')
    end.set(docx.oxml.ns.qn('w:name'), bookmark_name)
    tag.append(end)

def add_link(paragraph, link_to, text, tool_tip=None):
    # create hyperlink node
    hyperlink = docx.oxml.shared.OxmlElement('w:hyperlink')

    # set attribute for link to bookmark
    hyperlink.set(docx.oxml.shared.qn('w:anchor'), link_to,)

    if tool_tip is not None:
        # set attribute for link to bookmark
        hyperlink.set(docx.oxml.shared.qn('w:tooltip'), tool_tip,)

    new_run = docx.oxml.shared.OxmlElement('w:r')
    rPr = docx.oxml.shared.OxmlElement('w:rPr')
    new_run.append(rPr)
    new_run.text = text
    hyperlink.append(new_run)
    r = paragraph.add_run()
    r._r.append(hyperlink)
    r.font.name = "Calibri"
    # r.font.color.theme_color = MSO_THEME_COLOR_INDEX.HYPERLINK
    # r.font.underline = True

def set_n_matches_subtree(node_id, id_to_node, id_to_kids, id_to_n_matches_subtree):
    subtree_n_matches = int(id_to_node[node_id]['data']['n_matches'])

    if node_id in id_to_kids:
        for kid in id_to_kids[node_id]:
            subtree_n_matches += set_n_matches_subtree(kid, id_to_node, id_to_kids, id_to_n_matches_subtree)
    id_to_n_matches_subtree[node_id] = subtree_n_matches
    return subtree_n_matches

def save_hierarchical_graph_data_to_docx(graph_data, result_file, n_top_matches=None, sort_by_subtree=True):
    def get_hierarchical_bullets_aux(document, id_to_kids, id_to_node, id, tab, id_to_paragraph, id_to_n_matches_subtree, sort_by_subtree=True):
        bullet = '\u25E6' if tab % 2 == 1 else '\u2022'
        msg = f'{(" | " * tab)} {bullet} '

        p = document.add_paragraph(msg)
        id_to_paragraph[id] = p

        if id in id_to_kids:
            kids = id_to_kids[id]
            if sort_by_subtree:
                kids = sorted(kids, key=lambda n: int(id_to_n_matches_subtree[n]), reverse=True)
            else:
                kids = sorted(kids, key=lambda n: int(id_to_node[n]['data']['n_matches']), reverse=True)
            for k in kids:
                get_hierarchical_bullets_aux(document, id_to_kids, id_to_node, k, tab + 1, id_to_paragraph, id_to_n_matches_subtree, sort_by_subtree)


    def get_hierarchical_bullets(document, roots, id_to_kids, id_to_node, id_to_paragraph, id_to_n_matches_subtree, sort_by_subtree=True):
        tab = 0
        if sort_by_subtree:
            roots = sorted(roots, key=lambda n: int(id_to_n_matches_subtree[n]), reverse=True)
        else:
            roots = sorted(roots, key=lambda n: int(id_to_node[n]['data']['n_matches']), reverse=True)
        for root in roots:
            get_hierarchical_bullets_aux(document, id_to_kids, id_to_node, root, tab, id_to_paragraph, id_to_n_matches_subtree, sort_by_subtree=True)

    nodes = [d for d in graph_data if d['type'] == 'node']
    edges = [d for d in graph_data if d['type'] == 'edge']

    id_to_kids = create_dict_to_list([(e['data']['target'], e['data']['source']) for e in edges])
    nodes_ids = [n['data']['id'] for n in nodes]
    id_to_node = {d['data']['id']: d for d in nodes}

    all_kids = set()
    for id, kids in id_to_kids.items():
        all_kids = all_kids.union(kids)

    root_ids = set(nodes_ids).difference(all_kids)

    id_to_n_matches_subtree = {}
    for root in root_ids:
        set_n_matches_subtree(root, id_to_node, id_to_kids, id_to_n_matches_subtree)

    document = Document()
    document.add_heading('Key Point Analysis results', 0)
    document.add_heading('Hierarchical Key points:\n', 1)
    id_to_kids = create_dict_to_list([(e['data']['target'], e['data']['source']) for e in edges])

    logging.info('Creating key points hierarchy')

    id_to_paragraph1 = {}
    get_hierarchical_bullets(document, root_ids, id_to_kids, id_to_node, id_to_paragraph1, id_to_n_matches_subtree)

    logging.info('Creating key points matches tables')
    dicts, _ = read_dicts_from_csv(result_file)
    kp_to_dicts = create_dict_to_list([(d['kp'], d) for d in dicts])

    id_to_paragraph2 = {}
    if n_top_matches is None:
        document.add_heading(f'\n\nAll matches per key point:\n', 1)
    else:
        document.add_heading(f'\n\nTop {n_top_matches} matches per key point:\n', 1)

    for n in nodes:
        p = document.add_paragraph()
        id_to_paragraph2[n['data']["id"]] = p
        kp = n["data"]["kp"]
        p.add_run(f'\n\nKey point: {kp}  ({n["data"]["n_matches"]} matches)').bold = True

        records = []
        # for m in n['data']['matches']:
        #     records.append([m["sentence_text"], trunc_float(float(m["match_score"]), 3)])

        matches_dicts = kp_to_dicts[kp]
        if n_top_matches is not None and n_top_matches < len(kp_to_dicts[kp]):
            matches_dicts = matches_dicts[:n_top_matches]

        logging.info(f'creating table for KP: {kp}, n_matches: {len(matches_dicts)}')

        for d in matches_dicts:
            records.append([d["sentence_text"], trunc_float(float(d["match_score"]), 4)])

        table = document.add_table(rows=1, cols=2)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Sentence Text'
        hdr_cells[0].width = Inches(5)
        hdr_cells[1].text = 'Match Score'
        hdr_cells[1].width = Inches(0.5)
        for r in records:
            row_cells = table.add_row().cells
            row_cells[0].text = r[0]
            row_cells[0].width = Inches(5)
            row_cells[1].text = str(r[1])
            row_cells[1].width = Inches(0.5)

    # add a bookmark to every paragraph
    for id, paragraph in id_to_paragraph2.items():
        add_bookmark(paragraph=paragraph, bookmark_text="", bookmark_name=f'temp{id}')

    for id, paragraph in id_to_paragraph1.items():
        node = id_to_node[id]
        kp = node['data']['kp']
        n_matches = int(node["data"]["n_matches"])
        if n_matches == id_to_n_matches_subtree[id]:
            msg = f'{kp} ({n_matches} matches)'
        else:
            if sort_by_subtree:
                msg = f'{kp} ({id_to_n_matches_subtree[id]} matches in subtree, {n_matches} matches)'
            else:
                msg = f'{kp} ({n_matches} matches, {id_to_n_matches_subtree[id]} in subtree)'
        add_link(paragraph=paragraph, link_to=f'temp{id}', text=msg, tool_tip="click to see top sentences")

    out_file = result_file.replace('.csv', '_hierarchical.docx')
    logging.info(f'saving docx summary in file: {out_file}')
    document.save(out_file)
