def get_proposals_to_reject(used_process_executions_components, unused_process_executions_components: dict,
                            provision_proposals):
    if not unused_process_executions_components:
        proposals_to_reject = []
        return proposals_to_reject

    unused_process_executions_components = \
        set(list(unused_process_executions_components.keys())).difference(set(used_process_executions_components))

    proposals_to_reject = [proposal
                           for proposal, provider in provision_proposals
                           if proposal in unused_process_executions_components]

    return proposals_to_reject
