# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD 2-Clause License

"""ACE Action Frames.

The Action Frames implement the UMIM standard for frame processors.
See here to learn more about the UMIM standard:
https://docs.nvidia.com/ace/umim/latest/index.html

Note that at the moment the support for action frames is limited to the animation graph service.
More services and support for additional action frames will be added in the future.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from pipecat.frames.frames import ControlFrame, SystemFrame


def now_timestamp() -> datetime:
    """Helper to generate current timestamp."""
    return datetime.now(UTC)


def get_source_id() -> str:
    """Helper to generate a default source_id."""
    return "pipecat"


def new_uid() -> str:
    """Helper to create a new UID."""
    return str(uuid4())


@dataclass
class ActionFrame:
    """Frame that belongs to an action with a defined start and end.

    Args:
        action_id (str): A unique id for the action.
        parent_action_ids (list[str]): The ids of parent actions (causing this action)

    """

    action_id: str = field(kw_only=True, default_factory=new_uid)
    parent_action_ids: list[str] = field(kw_only=True, default_factory=list)


@dataclass
class BotActionFrame(ActionFrame):
    """Frame related to a bot action.

    Args:
        bot_id (Optional[str]):  An ID identifying the bot performing the action. This field is required if you
            support multi-bot interactions.
    """

    bot_id: str | None = field(kw_only=True, default=None)


@dataclass
class UserActionFrame(ActionFrame):
    """Frame related to a user action.

    Args:
        user_id (Optional[str]):  An ID identifying the user performing the action. This field is required if you
            support multi-user interactions.
    """

    user_id: str | None = field(kw_only=True, default=None)


@dataclass
class StartActionFrame(ControlFrame, ActionFrame):
    """Event to start an action.

    All other actions that can be started inherit from this base spec.
    The action_id is used to differentiate between multiple runs of the same action.
    """


@dataclass
class StartedActionFrame(ControlFrame, ActionFrame):
    """The execution of an action has started.

    Args:
        action_started_at (datetime): The timestamp of when the action has started.

    """

    action_id: str = field(kw_only=True)
    action_started_at: datetime = field(kw_only=True, default_factory=now_timestamp)


@dataclass
class StopActionFrame(ControlFrame, ActionFrame):
    """An action needs to be stopped.

    This should be used to proactively stop an action that can take a
    longer period of time, e.g., a gesture.
    """

    action_id: str = field(kw_only=True)


@dataclass
class ChangeActionFrame(ControlFrame, ActionFrame):
    """The parameters of a running action needs to be changed.

    Updating running actions is useful for longer running
    actions (e.g. an avatar animation) which can adapt their behavior dynamically. For example, a nodding animation
    can change its speed depending on the voice activity level.
    """

    action_id: str = field(kw_only=True)


@dataclass
class UpdatedActionFrame(ControlFrame, ActionFrame):
    """A running action provides a (partial) result.

    Ongoing actions can provide partial updates on the current status
    of the action. An ActionUpdated should always update the payload of the action object and provide
    the type of update.

    Args:
        action_updated_at (datetime): The timestamp of when the action was updated. The timestamp should represent
            the system time the action actually changed, not the timestamp of when the `Updated` event was created
            (for this, there is the `event_created_at` field).

    """

    action_updated_at: datetime = field(kw_only=True, default_factory=now_timestamp)


@dataclass
class FinishedActionFrame(ControlFrame, ActionFrame):
    """An action has finished its execution.

    An action can finish either because the action has completed or
    failed (natural completion) or it can finish because it was stopped by the IM. The success (or failure) of the
    execution is marked using the status_code attribute.

    Args:
        action_finished_at (datetime): The timestamp of when the action has finished.
        is_success (bool): Did the action finish successfully
        was_stopped (Optional[bool]): Was the action stopped by a Stop event
        failure_reason (Optional[str]): Reason for action failure in case the action did not execute successfully
    """

    action_id: str = field(kw_only=True)
    action_finished_at: datetime = field(kw_only=True, default_factory=now_timestamp)
    is_success: bool = field(kw_only=True, default=True)
    was_stopped: bool | None = field(kw_only=True, default=None)
    failure_reason: str | None = field(kw_only=True, default=None)


# Facial Gesture Bot Action


@dataclass
class StartFacialGestureBotActionFrame(StartActionFrame, BotActionFrame):
    """The bot should start making a facial gesture.

    Args:
        facial_gesture: Natural language description (NLD) of the facial gesture
            or expression. Availability of facial gestures depends on the
            interactive system.

            Minimal NLD set:
            The following gestures should be supported by every interactive system
            implementing this action:
            - smile
            - laugh
            - frown
            - wink
    """

    facial_gesture: str


@dataclass
class StartedFacialGestureBotActionFrame(StartedActionFrame, BotActionFrame):
    """The bot has started to perform the facial gesture."""


@dataclass
class StopFacialGestureBotActionFrame(StopActionFrame, BotActionFrame):
    """Stop the facial gesture or expression.

    All gestures have a limited lifetime and finish on "their own"
    (e.g., in an interactive avatar system a "smile" gesture could be implemented by a 1 second animation clip where
    some facial bones are animated). The IM can use this action to stop an expression
    before it would be naturally done.
    """


@dataclass
class FinishedFacialGestureBotActionFrame(FinishedActionFrame, BotActionFrame):
    """The facial gesture was performed."""


# Gesture Bot Action


@dataclass
class StartGestureBotActionFrame(StartActionFrame, BotActionFrame):
    """The bot should start making a specific gesture.

    Args:
        gesture: Natural language description (NLD) of the gesture. Availability of gestures depends on the
            interaction system. If a system supports this action, the following base gestures need to be supported:
            `affirm`, `negate`, `attract`
    """

    gesture: str


@dataclass
class StartedGestureBotActionFrame(StartedActionFrame, BotActionFrame):
    """The bot has started to perform the gesture."""


@dataclass
class StopGestureBotActionFrame(StopActionFrame, BotActionFrame):
    """Stop the gesture.

    All gestures have a limited lifetime and finish on 'their own'. Gesture are meant to accentuate
    a certain situation or statement. For example, in an interactive avatar system a `affirm` gesture could be
    implemented by a 1 second animation clip where the avatar nods twice.
    The IM can use this action to stop a gesture before it would be naturally done.
    """


@dataclass
class FinishedGestureBotActionFrame(FinishedActionFrame, BotActionFrame):
    """The gesture was performed."""


# Posture Bot Action


@dataclass
class StartPostureBotActionFrame(StartActionFrame, BotActionFrame):
    """The bot should start adopting the specified posture.

    Args:
        posture (str) : Natural language description (NLD) of the posture. The availability of postures depends on the
            interaction system. Postures should be expressed hierarchically such that interactive systems that provide
            less nuanced postures can fall back onto higher level postures.
            The following base postures need to be supported
            by all interactive systems supporting this action.: "idle", "attentive"
    """

    posture: str


@dataclass
class StartedPostureBotActionFrame(StartedActionFrame, BotActionFrame):
    """The bot has attained the posture."""


@dataclass
class StopPostureBotActionFrame(StopActionFrame, BotActionFrame):
    """Stop the posture.

    Postures have no lifetime, so unless the IM calls the Stop action the bot will
    keep the posture indefinitely.
    """


@dataclass
class FinishedPostureBotActionFrame(FinishedActionFrame, BotActionFrame):
    """The posture was stopped."""


# Position Bot Action


@dataclass
class StartPositionBotActionFrame(StartActionFrame, BotActionFrame):
    """The bot needs to hold a new position.

    Args:
        position (str) : Specify the position the bot needs to move to and maintain.
            Availability of positions depends on the interactive system. Positions are typically
            structured hierarchically into base position and position modifiers ("off center").

            Minimal NLD set:
            The following base positions are supported by all interactive systems (that support this action):
            center : Default position of the bot
            left : Bot should be positioned to the left (from the point of view of the bot)
            right: Bot should be positioned to the right (from the point of view of the bot)
    """

    position: str


@dataclass
class StartedPositionBotActionFrame(StartedActionFrame, BotActionFrame):
    """The bot has started to transition to the new position."""


@dataclass
class UpdatedPositionBotActionFrame(UpdatedActionFrame, BotActionFrame):
    """The bot has arrived at the position and is maintaining that position for the entire action duration.

    Args:
        position_reached (str) : The position the bot has reached.
    """

    position_reached: str


@dataclass
class StopPositionBotActionFrame(StopActionFrame, BotActionFrame):
    """Stop holding the position.

    The bot will return to the position it had before the call.
    Position holding actions have an infinite lifetime, so unless the IM calls the Stop action the bot maintains
    the position indefinitely.  Alternatively PositionBotAction actions can be overwritten, since the modality policy
    is Override.
    """


@dataclass
class FinishedPositionBotActionFrame(FinishedActionFrame, BotActionFrame):
    """The bot shifted back to the original position before this action.

    This might be a neutral position or the position of any PositionBotAction overwritten
    by this action that now gains the "focus".
    """


# Presence User Action
@dataclass
class StartedPresenceUserActionFrame(StartedActionFrame, UserActionFrame, SystemFrame):
    """The interactive system detects the presence of a user in the system.

    TODO: We inherit from SystemFrame to circumvent the frame deletion issue with StartInterruptionFrame.
    This is a temporary fix only and needs to be reconsidered once the action concept is properly
    introduced.
    """


@dataclass
class FinishedPresenceUserActionFrame(FinishedActionFrame, UserActionFrame, SystemFrame):
    """The interactive system detects the user's absence.

    TODO: We inherit from SystemFrame to circumvent the frame deletion issue with StartInterruptionFrame.
    This is a temporary fix only and needs to be reconsidered once the action concept is properly
    introduced.
    """


# Attention User Action
@dataclass
class StartedAttentionUserActionFrame(StartedActionFrame, UserActionFrame):
    """The interactive system detects some level of engagement of the user.

    Args:
        attention_level (str): Attention level. Minimal supported values are "engaged" and "disengaged".
            Many systems support more granular levels, such as "engaged, partially" "disengaged, looking at phone
    """

    attention_level: str


@dataclass
class UpdatedAttentionUserActionFrame(UpdatedActionFrame, UserActionFrame):
    """The interactive system provides an update to the engagement level.

    Args:
        attention_level (str): Attention level. Minimal supported values are "engaged" and "disengaged".
            Many systems support more granular levels, such as "engaged, partially" "disengaged, looking at phone
    """

    attention_level: str


@dataclass
class FinishedAttentionUserActionFrame(FinishedActionFrame, UserActionFrame):
    """The system detects the user to be disengaged with the interactive system."""


# Motion Effect Camera Action
@dataclass
class StartMotionEffectCameraActionFrame(StartActionFrame, BotActionFrame):
    """Perform the described camera motion effect.

    Args:
        effect (str): Natural language description (NLD) of the effect. Availability of effects depends
            on the interactive system. Minimal NLD set:
            The following camera effects should be supported by every interactive system implementing this action:
            shake, jump cut in, jump cut out
    """

    effect: str


@dataclass
class StartedMotionEffectCameraActionFrame(StartedActionFrame, BotActionFrame):
    """Camera effect started."""


@dataclass
class StopMotionEffectCameraActionFrame(StopActionFrame, BotActionFrame):
    """Stop the camera effect.

    All effects have a limited lifetime and finish "on their own"
    (e.g., in an interactive avatar system a "shake" effect could be implemented by a 1 second camera motion).
    The IM can use this action to stop a camera effect before it would be naturally done.
    """


@dataclass
class FinishedMotionEffectCameraActionFrame(FinishedActionFrame, BotActionFrame):
    """Camera effect finished."""


# Shot Camera Action
@dataclass
class StartShotCameraActionFrame(StartActionFrame, BotActionFrame):
    """Start a new shot.

    Args:
        shot (str): Natural language description (NLD) of the shot. Availability of shots depends on
            the interactive system. Minimal NLD set:
            The following shots should be supported by every interactive system implementing this action:
                full, medium, close-up
        start_transition (str): NLD of the transition to the new shot. This should describe the movement.
            Minimal NLD set:
            The following shots should be supported by every interactive system implementing this action: cut, dolly
    """

    shot: str
    start_transition: str


@dataclass
class StartedShotCameraActionFrame(StartedActionFrame, BotActionFrame):
    """The camera shot started."""


@dataclass
class StopShotCameraActionFrame(StopActionFrame, BotActionFrame):
    """Stop the camera shot.

    The camera will return to the shot it had before this action was started.
    ShotCameraAction actions have an infinite lifetime, so unless the IM calls the Stop action the
    camera maintains the shot indefinitely.

    Args:
        stop_transition (str): NLD of the transition back to the previous shot (override modality).
            This should describe the movement.
            Minimal NLD set:
            The following shots should be supported by every interactive system implementing this action: cut, dolly
    """

    stop_transition: str


@dataclass
class FinishedShotCameraActionFrame(FinishedActionFrame, BotActionFrame):
    """The camera shot was stopped.

    The camera has returned to the shot it had before (either a neutral shot) or the shot specified by any
    overwritten ShotCameraAction actions.
    """
