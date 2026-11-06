"""Read-only handoff diagnosis; no network, model import or dataset decoding."""
from pathlib import Path
from applications.rag.junior_transport_v001 import member, digest


def local_models(campaign, root):
    """Compare downloaded mirrors with pinned bytes, never trust 'master' alone."""
    root = Path(root)
    release = campaign['populations']['reproduction']
    groups = {}
    for kind, prefix in release['models'].items():
        records = [r for r in release['files'] if r['path'].startswith(prefix+'/')]
        rows = []
        for record in records:
            path = member(root, record['path'])
            if not path.is_file():
                status = 'MISSING'
            elif path.stat().st_size != record['size']:
                status = 'SIZE_MISMATCH'
            elif digest(path) != record['sha256']:
                status = 'SHA256_MISMATCH'
            else:
                status = 'MATCH'
            rows.append(dict(path=record['path'], status=status))
        groups[kind] = dict(matches=bool(rows) and all(r['status']=='MATCH' for r in rows),
                            expected_files=len(rows), files=rows)
    return dict(root=str(root), models=groups,
                all_pinned_model_bytes_match=bool(groups) and all(g['matches'] for g in groups.values()),
                model_loaded=False, downloaded=False,
                note='Matching model files can be reused; this does not supply a qualified dataset release.')


def diagnosis(report, *, access_config_present=False):
    return dict(schema='qduet.junior-readiness-diagnosis.v1',
        category='READY_FOR_PREPARATION' if report['ready'] else 'DATA_RELEASE_NOT_READY',
        ready=report['ready'], population=report['population'], datasets=report['datasets'],
        releases=report['readiness'],
        release_ready_is_service_health=False, private_access_config_present=access_config_present,
        private_service_contacted=False, local_raw_data_auto_converted=False,
        raw_questions_and_answers_are_passage_labels=False,
        can_skip_release_gate=False,
        responsible_party='Project owner prepares and registers the missing qualified releases; junior runs the frozen suite.',
        owner_required=['Original Top50 inputs and fixed role/component identities',
                        'Reviewed visible-text training labels and separately sealed evaluation labels',
                        'Question-overlap, label-view and oracle/headroom evidence',
                        'Archive/member size-SHA inventory and private release registration'],
        junior_action='Keep existing downloads. Do not rebuild retrieval, relabel or change ready flags to bypass this guard.',
        available_exposed_reproduction=dict(dataset='triviaqa',train=512,evaluation=128,
            preparation_command='bash scripts/run_two_datasets.sh --datasets triviaqa --population reproduction --prepare-only',
            formal=False, current_service_health='NOT_CHECKED'),
        model_check_command='bash scripts/run_two_datasets.sh --diagnose --check-local-models .',
        explanation='docs/junior-startup-data-release-faq.md',
        scientific_execution=False, downloads_performed=False)


def release_error(report):
    reasons=' | '.join(r['dataset']+': '+r['reason'] for r in report['readiness'] if not r['ready'])
    return ('DATA_RELEASE_NOT_READY：正式输入/段落标签发布未完成；尚未运行模型。'
            '原始问答、wiki_dpr/FAISS和模型下载不能替代正式release。'
            '请保留已下载文件，不要重复下载或把status改为ready。'
            '该缺口需项目负责人补齐，师弟不必自行重建实验。'
            '诊断：bash scripts/run_two_datasets.sh --diagnose；'
            '说明：docs/junior-startup-data-release-faq.md\n'+reasons)
