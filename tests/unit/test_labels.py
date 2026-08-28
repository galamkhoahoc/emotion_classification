"""
Unit tests for label conversion utilities.
"""

import pytest
import torch
from src.utils.labels import convert_label_to_tensor, convert_binary_vector_to_labels


class TestConvertLabelToTensor:
    def test_multiclass_int_to_tensor(self):
        tensor = convert_label_to_tensor(2, num_labels=5, problem_type="multiclass_classification")
        assert tensor.dtype == torch.long
        assert tensor.item() == 2

    def test_multiclass_string_int_to_tensor(self):
        tensor = convert_label_to_tensor("3", num_labels=5, problem_type="multiclass_classification")
        assert tensor.dtype == torch.long
        assert tensor.item() == 3

    def test_multiclass_invalid_type(self):
        with pytest.raises(TypeError, match="Label must be an integer"):
            convert_label_to_tensor([1, 2], num_labels=5, problem_type="multiclass_classification")

    def test_multiclass_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            convert_label_to_tensor(5, num_labels=5, problem_type="multiclass_classification")
        with pytest.raises(ValueError, match="out of range"):
            convert_label_to_tensor(-1, num_labels=5, problem_type="multiclass_classification")

    def test_multilabel_list_to_binary_vector(self):
        tensor = convert_label_to_tensor([0, 2, 4], num_labels=5, problem_type="multilabel_classification")
        assert tensor.dtype == torch.float32
        assert tensor.shape == (5,)
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        assert torch.all(tensor == expected)

    def test_multilabel_empty_list_rejection(self):
        with pytest.raises(ValueError, match="at least one label"):
            convert_label_to_tensor([], num_labels=5, problem_type="multilabel_classification")

    def test_multilabel_invalid_type(self):
        with pytest.raises(TypeError, match="Label must be a list/tuple"):
            convert_label_to_tensor(2, num_labels=5, problem_type="multilabel_classification")

    def test_multilabel_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            convert_label_to_tensor([0, 5], num_labels=5, problem_type="multilabel_classification")


class TestConvertBinaryVectorToLabels:
    def test_binary_vector_to_emotion_names(self):
        emotion_labels = ["A", "B", "C", "D", "E"]
        binary_vector = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        names = convert_binary_vector_to_labels(binary_vector, emotion_labels)
        assert names == ["A", "C"]

    def test_binary_vector_list_input(self):
        emotion_labels = ["A", "B", "C", "D", "E"]
        binary_vector = [1.0, 0.0, 1.0, 0.0, 0.0]
        names = convert_binary_vector_to_labels(binary_vector, emotion_labels)
        assert names == ["A", "C"]

    def test_length_mismatch(self):
        emotion_labels = ["A", "B", "C"]
        binary_vector = [1.0, 0.0]
        with pytest.raises(ValueError, match="does not match emotion labels"):
            convert_binary_vector_to_labels(binary_vector, emotion_labels)

    def test_round_trip_property(self):
        emotion_labels = ["A", "B", "C", "D", "E"]
        original_indices = [1, 3]
        
        # Convert to tensor
        tensor = convert_label_to_tensor(original_indices, num_labels=5, problem_type="multilabel_classification")
        
        # Convert back to names
        names = convert_binary_vector_to_labels(tensor, emotion_labels)
        
        # Verify names match expected names from original indices
        expected_names = [emotion_labels[i] for i in original_indices]
        assert names == expected_names
