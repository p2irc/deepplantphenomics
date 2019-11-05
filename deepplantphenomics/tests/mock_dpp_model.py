from deepplantphenomics.deepplantpheno import DPPModel


class MockDPPModel(DPPModel):
    def _graph_problem_loss(self, pred, lab):
        pass

    def _assemble_graph(self):
        pass

    def compute_full_test_accuracy(self):
        pass

    def forward_pass_with_file_inputs(self, x):
        pass

    def forward_pass_with_interpreted_outputs(self, x):
        pass

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        pass
