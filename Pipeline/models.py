from transformers import BartTokenizer, BartForConditionalGeneration
class Model:
    """
    Base Model class with a factory method to return the appropriate model object.
    """

    def classify_text(self, text):
        """
        Abstract method to classify text.
        This should be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement this method")

    @staticmethod
    def get_model(name, label_options, multi_class):
        """
        :param name: the name of the model
        :return: the model object
        """
        if name.lower() == 'bart':
            return Bart(label_options, multi_class)
        else:
            raise ValueError(f"Model '{name}' is not available")

    def extract_labels_from_generated_text(self, generated_text, label_options):
        relevant_labels = []
        for label in label_options:
            if label.lower() in generated_text.lower():
                relevant_labels.append(label)
        return relevant_labels


class Bart(Model):
    """
    The BART model
    """

    def __init__(self, label_options, multi_class=False):
        self.label_options = label_options
        self.multi_class = multi_class
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    def extract_labels_from_generated_text(self, generated_text, label_options):
        return super().extract_labels_from_generated_text(generated_text, label_options)

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=100, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def classify_text(self, text):
        """
        :param text: the text that needs to be classified
        :return: a list of all the labels corresponding to the given text
        """
        prompt = text + "<|endoftext|>" + "Question: Which of the following labels apply?" + self.label_options + "Answer: "
        generated_text = self.generate_text(prompt)
        return self.extract_labels_from_generated_text(generated_text, self.label_options)
