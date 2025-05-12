import asyncio
from validation_note_agent import ConsistencyVerifierAgent
import nest_asyncio
import os

output1 = """ #Understanding the French Revolution

The French Revolution was a pivotal period in history that transformed France from a monarchy into a republic. It began in 1789 due to a combination of social, political, and economic factors that the existing regime could not address. Key events, such as the convocation of the Estates General and the Storming of the Bastille, marked the revolution's beginning. The revolution led to significant changes, including the abolition of feudalism and the establishment of the First Republic. It was characterized by struggles for political control, the execution of King Louis XVI, and the Reign of Terror, before ultimately leading to the rise of Napoleon Bonaparte.

## Key points:
- The French Revolution began in 1789 and ended in 1799 with Napoleon's rise to power.
- It was sparked by financial crises and social discontent under the ancien régime.
- Key events include the convening of the Estates General, the National Assembly's formation, and the Storming of the Bastille.
- The monarchy was abolished, and the First Republic was established after the execution of King Louis XVI.
- The Reign of Terror, led by Robespierre, resulted in thousands of executions.
- The revolution ended with the coup of 18 Brumaire, leading to the Consulate and Napoleon as First Consul.)
"""
output2 =  """1.⁠ ⁠*Diffusion Models*: 
   - *Explanation*: Diffusion models are a class of generative models that transform data through a series of latent variables by progressively adding noise until the data becomes indistinguishable from random noise. The reverse process, which aims to denoise the data, is modeled using deep learning techniques, typically approximating the denoising function with a normal distribution. The training of these models is guided by the evidence lower bound (ELBO), which leads to a least-squares formulation.
   - *Importance*: Diffusion models have gained prominence due to their ability to generate high-quality data, particularly in image synthesis. They represent a significant advancement in generative modeling, providing a robust framework for creating realistic outputs.

2.⁠ ⁠*Classifier-Free Guidance*: 
   - *Explanation*: This technique enhances the generative capabilities of diffusion models by integrating class information directly into the model, eliminating the need for a separate classifier. It involves embedding class information into the U-Net architecture, similar to how time steps are incorporated. This allows the model to generate outputs that are both conditional on class information and unconditional, thus offering greater flexibility.
   - *Importance*: Classifier-free guidance improves the quality and diversity of generated outputs, making it a crucial advancement in the field of generative modeling. It streamlines the generation process and enhances the model's ability to produce high-fidelity results.

3.⁠ ⁠*Cascaded Conditional Generation*: 
   - *Explanation*: This approach utilizes a series of diffusion models to generate images at progressively higher resolutions. It starts with a low-resolution image and employs subsequent models to refine and enhance the image quality based on the previous outputs and additional conditioning information, such as class labels or textual descriptions.
   - *Importance*: Cascaded conditional generation is vital for producing high-quality, detailed images. It allows for a structured approach to image synthesis, ensuring that the final outputs are not only high-resolution but also coherent and contextually relevant.

4.⁠ ⁠*Improving Generation Quality*: 
   - *Explanation*: Various enhancements to diffusion models focus on improving the quality of generated images. These include estimating the variances of the reverse process, adjusting the noise schedule, and employing cascaded models for high-resolution outputs. Each of these techniques contributes to refining the generative process.
   - *Importance*: Enhancing generation quality is essential for practical applications of generative models, particularly in fields like art, design, and entertainment. Higher quality outputs lead to more realistic and diverse images, which are crucial for user satisfaction and engagement.

5.⁠ ⁠*Stochastic Nature and Diversity*: 
   - *Explanation*: The stochastic nature of diffusion models allows them to generate multiple diverse outputs from the same input conditions, such as a text prompt. This randomness is a key feature that enables the creation of varied images while still adhering to the specified constraints.
   - *Importance*: The ability to produce diverse outputs is particularly beneficial in creative applications, where variation is often desired. This characteristic enhances the utility of diffusion models in generating unique and innovative content, making them valuable tools in various domains.)
"""

api_key = os.getenv("OPENAI_API_KEY")


verifier = ConsistencyVerifierAgent(openai_api_key=api_key)
merged_note = asyncio.run(verifier.run(note_1=output1, note_2=output2))
print(merged_note)