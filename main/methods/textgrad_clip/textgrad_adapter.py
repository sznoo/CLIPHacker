# methods/textgrad_clip/textgrad_adapter.py

import textgrad as tg

from methods.textgrad_clip.configs import PROMPT_ROLE, TEXTGRAD_CONSTRAINTS


def make_prompt_variable(prompt: str) -> tg.Variable:
    return tg.Variable(
        prompt,
        requires_grad=True,
        role_description=PROMPT_ROLE,
    )


def make_optimizer(prompt_var: tg.Variable, gradient_memory: int = 0):
    return tg.TGD(
        parameters=[prompt_var],
        constraints=TEXTGRAD_CONSTRAINTS,
        gradient_memory=gradient_memory,
    )


def run_textgrad_step(
    prompt_var: tg.Variable,
    optimizer,
    feedback_instruction: str,
):
    optimizer.zero_grad()

    loss_fn = tg.TextLoss(feedback_instruction)
    loss = loss_fn(prompt_var)
    loss.backward()
    optimizer.step()

    return prompt_var.get_value(), loss.get_value()