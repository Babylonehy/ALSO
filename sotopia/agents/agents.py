
class BaseAgent(Generic[ObsType, ActType], MessengerMixin):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> None:
        MessengerMixin.__init__(self)
        if agent_profile is not None:
            self.profile = agent_profile
            self.agent_name = self.profile.first_name + " " + self.profile.last_name
        elif uuid_str is not None:
            # try retrieving profile from database
            try:
                self.profile = AgentProfile.get(pk=uuid_str)
            except NotFoundError:
                raise ValueError(f"Agent with uuid {uuid_str} not found in database")
            self.agent_name = self.profile.first_name + " " + self.profile.last_name
        else:
            assert (
                agent_name is not None
            ), "Either agent_name or uuid_str must be provided"
            self.agent_name = agent_name

        self._goal: str | None = None
        self.model_name: str = ""

    @property
    def goal(self) -> str:
        assert self._goal is not None, "attribute goal has to be set before use"
        return self._goal

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    async def aact(self, obs: ObsType) -> ActType:
        raise NotImplementedError

    def reset(self) -> None:
        self.reset_inbox()


class LLMAgent(BaseAgent[Observation, AgentAction]):
    def __init__(
        self,
        agent_name: str | None = None,
        uuid_str: str | None = None,
        agent_profile: AgentProfile | None = None,
        model_name: str = "gpt-4o-mini",
        script_like: bool = False,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            agent_name=agent_name,
            uuid_str=uuid_str,
            agent_profile=agent_profile,
        )
        self.model_name = model_name
        self.script_like = script_like
        self.max_tokens = max_tokens

    @property
    def goal(self) -> str:
        if self._goal is not None:
            return self._goal
        else:
            raise Exception("Goal is not set.")

    @goal.setter
    def goal(self, goal: str) -> None:
        self._goal = goal

    def act(
        self,
        _obs: Observation,
    ) -> AgentAction:
        raise Exception("Sync act method is deprecated. Use aact instead.")

    async def aact(self, obs: Observation) -> AgentAction:
        self.recv_message("Environment", obs)

        if self._goal is None:
            self._goal = await agenerate_goal(
                self.model_name,
                background=self.inbox[0][
                    1
                ].to_natural_language(),  # Only consider the first message for now
            )
        history = "\n".join(f"{y.to_natural_language()}" for x, y in self.inbox)
        # for debug
        # rich.print(Panel(history, title="Action Generate History"))

        if len(obs.available_actions) == 1 and "none" in obs.available_actions:
            return AgentAction(action_type="none", argument="")

        else:
            action = await agenerate_action(
                self.model_name,
                history=history,
                turn_number=obs.turn_number,
                action_types=obs.available_actions,
                agent=self.agent_name,
                goal=self.goal,
                max_tokens=self.max_tokens,
                script_like=self.script_like,
            )
            # Temporary fix for mixtral-moe model for incorrect generation format
            if "Mixtral-8x7B-Instruct-v0.1" in self.model_name:
                current_agent = self.agent_name
                if f"{current_agent}:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(f"{current_agent}: ", "")
                elif f"{current_agent} said:" in action.argument:
                    print("Fixing Mixtral's generation format")
                    action.argument = action.argument.replace(
                        f"{current_agent} said: ", ""
                    )

            return action
