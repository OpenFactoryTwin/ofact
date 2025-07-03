import unittest

import ofact.twin.state_model.basic_elements as be
import ofact.twin.state_model.entities as ent
import ofact.twin.state_model.processes as pro


class TestDigitalTwinObject(unittest.TestCase):

    def test_next_id(self):
        id_ = be.DigitalTwinObject(None).identification
        new_id = be.DigitalTwinObject(None).identification
        self.assertEqual(id_ + 1, new_id)


class TestTransformationModel(unittest.TestCase):

    def test_get_transformed_parts_assembly(self):
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")
        sub_part_entity_type = ent.EntityType(identification=None,
                                              name="sub part")
        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        sub_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=sub_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.SUB_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)

        main_part_part_transformation_node.add_child(output_part_part_transformation_node)
        sub_part_part_transformation_node.add_child(output_part_part_transformation_node)
        output_part_part_transformation_node.add_parent(main_part_part_transformation_node,
                                                        sub_part_part_transformation_node)

        assembly_transformation_model = \
            pro.TransformationModel(identification=None,
                                    root_nodes=[main_part_part_transformation_node, sub_part_part_transformation_node])

        main_part_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=main_part_entity_type)
        sub_part_part = ent.Part(identification=None,
                                 name="Sub Part",
                                 entity_type=sub_part_entity_type)

        transformed_parts, destroyed_parts = assembly_transformation_model.get_transformed_entities(
            input_parts=[main_part_part, sub_part_part], input_resources=[])
        transformed_part = transformed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type,
                                     parts_removable=[True])
        right_sub_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=sub_part_entity_type,
                                  part_of=right_output_part)
        right_output_part.parts = [right_sub_part]

        self.assertEqual(transformed_part.entity_type, right_output_part.entity_type,
                         "ASSEMBLY Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part.parts_removable, right_output_part.parts_removable,
                         "ASSEMBLY Incorrect MAIN_PART parts_removable")
        self.assertEqual(transformed_part.parts[0].entity_type, right_output_part.parts[0].entity_type,
                         "ASSEMBLY Incorrect MAIN_PART SUB_ENTITY EntityType")
        self.assertEqual(transformed_part.parts[0].part_of.entity_type, right_output_part.parts[0].part_of.entity_type,
                         "ASSEMBLY Incorrect MAIN_PART SUB_ENTITY MAIN_PART EntityType")

    def test_get_transformed_parts_blank(self):
        blank_entity_type = ent.EntityType(identification=None,
                                           name="blank")
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")

        blank_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=blank_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.BLANK,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)

        blank_part_transformation_node.add_child(output_part_part_transformation_node)
        output_part_part_transformation_node.add_parent(blank_part_transformation_node)

        blank_transformation_model = \
            pro.TransformationModel(identification=None,
                                    root_nodes=[blank_part_transformation_node])

        blank_part = ent.Part(identification=None,
                              name="Blank",
                              entity_type=blank_entity_type)

        transformed_parts, destroyed_parts = \
            blank_transformation_model.get_transformed_entities(input_parts=[blank_part],
                                                                input_resources=[])
        transformed_part = transformed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type,
                                     parts_removable=[False])
        right_blank = ent.Part(identification=None,
                               name="Main Part",
                               entity_type=blank_entity_type,
                               part_of=right_output_part)

        right_output_part.parts = [right_blank]

        self.assertEqual(transformed_part.entity_type, right_output_part.entity_type,
                         "BLANK Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part.parts_removable, right_output_part.parts_removable,
                         "BLANK Incorrect MAIN_PART parts_removable")
        self.assertEqual(transformed_part.parts[0].entity_type, right_output_part.parts[0].entity_type,
                         "BLANK Incorrect MAIN_PART SUB_ENTITY EntityType")
        self.assertEqual(transformed_part.parts[0].part_of.entity_type,
                         right_output_part.parts[0].part_of.entity_type,
                         "BLANK Incorrect MAIN_PART SUB_ENTITY MAIN_PART EntityType")

    def test_get_transformed_parts_ingredient(self):
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")
        ingredient_entity_type = ent.EntityType(identification=None,
                                                name="ingredient")
        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        ingredient_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=ingredient_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.INGREDIENT,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)

        main_part_part_transformation_node.add_child(output_part_part_transformation_node)
        ingredient_part_transformation_node.add_child(output_part_part_transformation_node)
        output_part_part_transformation_node.add_parent(main_part_part_transformation_node,
                                                        ingredient_part_transformation_node)

        ingredient_transformation_model = \
            pro.TransformationModel(identification=None,
                                    root_nodes=[main_part_part_transformation_node,
                                                ingredient_part_transformation_node])

        main_part_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=main_part_entity_type)
        ingredient_part = ent.Part(identification=None,
                                   name="Ingredient",
                                   entity_type=ingredient_entity_type)

        transformed_parts, destroyed_parts = \
            ingredient_transformation_model.get_transformed_entities(input_parts=[main_part_part, ingredient_part],
                                                                     input_resources=[])
        transformed_part = transformed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type,
                                     parts_removable=[False])
        right_ingredient = ent.Part(identification=None,
                                    name="Ingredient",
                                    entity_type=ingredient_entity_type,
                                    part_of=right_output_part)
        right_output_part.parts = [right_ingredient]

        self.assertEqual(transformed_part.entity_type, right_output_part.entity_type,
                         "INGREDIENT Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part.parts_removable, right_output_part.parts_removable,
                         "INGREDIENT Incorrect MAIN_PART parts_removable")
        self.assertEqual(transformed_part.parts[0].entity_type, right_output_part.parts[0].entity_type,
                         "INGREDIENT Incorrect MAIN_PART SUB_ENTITY EntityType")
        self.assertEqual(transformed_part.parts[0].part_of.entity_type, right_output_part.parts[0].part_of.entity_type,
                         "INGREDIENT Incorrect MAIN_PART SUB_ENTITY MAIN_PART EntityType")

    def test_get_transformed_parts_disassembly(self):
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")
        sub_part_entity_type = ent.EntityType(identification=None,
                                              name="sub part")
        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_part_transformation_node1 = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_part_transformation_node2 = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=sub_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.DISASSEMBLE,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)

        main_part_part_transformation_node.add_child(output_part_part_transformation_node1,
                                                     output_part_part_transformation_node2)
        output_part_part_transformation_node1.add_parent(main_part_part_transformation_node)
        output_part_part_transformation_node2.add_parent(main_part_part_transformation_node)

        disassembly_transformation_model = \
            pro.TransformationModel(identification=None,
                                    root_nodes=[main_part_part_transformation_node])

        main_part_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=main_part_entity_type,
                                  part_of=None,
                                  parts_removable=[])
        sub_part_part = ent.Part(identification=None,
                                 name="Sub Part",
                                 entity_type=sub_part_entity_type,
                                 part_of=main_part_part,
                                 parts=[],
                                 parts_removable=[])
        main_part_part.parts.append(sub_part_part)
        main_part_part.parts_removable.append(True)

        transformed_parts, destroyed_parts = \
            disassembly_transformation_model.get_transformed_entities(input_parts=[main_part_part],
                                                                      input_resources=[])

        transformed_part1 = transformed_parts[0][0]
        transformed_part2 = transformed_parts[1][0]

        right_output_part1 = ent.Part(identification=None,
                                      name="Main Part",
                                      entity_type=main_part_entity_type,
                                      part_of=None,
                                      parts=[],
                                      parts_removable=[])
        right_output_part2 = ent.Part(identification=None,
                                      name="Sub Part",
                                      entity_type=sub_part_entity_type,
                                      part_of=None,
                                      parts=[],
                                      parts_removable=[])

        self.assertEqual(transformed_part1.entity_type, right_output_part1.entity_type,
                         "DISASSEMBLY Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part1.part_of, right_output_part1.part_of,
                         "DISASSEMBLY Incorrect MAIN_PART part_of")
        self.assertEqual(transformed_part1.parts, right_output_part1.parts,
                         "DISASSEMBLY Incorrect MAIN_PART parts")
        self.assertEqual(transformed_part1.parts_removable, right_output_part1.parts_removable,
                         "DISASSEMBLY Incorrect MAIN_PART parts_removable")

        self.assertEqual(transformed_part2.entity_type, right_output_part2.entity_type,
                         "DISASSEMBLY Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part2.part_of, right_output_part2.part_of,
                         "DISASSEMBLY Incorrect MAIN_PART part_of")
        self.assertEqual(transformed_part2.parts, right_output_part2.parts,
                         "DISASSEMBLY Incorrect MAIN_PART parts")
        self.assertEqual(transformed_part2.parts_removable, right_output_part2.parts_removable,
                         "DISASSEMBLY Incorrect MAIN_PART parts_removable")

    def test_get_transformed_parts_destroy(self):
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")

        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.DESTROYED)

        destroying_transformation_model = \
            pro.TransformationModel(identification=None, root_nodes=[main_part_part_transformation_node])

        main_part_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=main_part_entity_type)

        transformed_parts, destroyed_parts = \
            destroying_transformation_model.get_transformed_entities(input_parts=[main_part_part],
                                                                     input_resources=[])

        destroyed_part = destroyed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type,
                                     part_of=None,
                                     parts=[],
                                     parts_removable=[])

        self.assertEqual(destroyed_part.entity_type, right_output_part.entity_type,
                         "DESTROYING Incorrect MAIN_PART EntityType")
        self.assertEqual(destroyed_part.part_of, right_output_part.part_of,
                         "DESTROYING Incorrect MAIN_PART part_of")
        self.assertEqual(destroyed_part.parts, right_output_part.parts,
                         "DESTROYING Incorrect MAIN_PART parts")
        self.assertEqual(destroyed_part.parts_removable, right_output_part.parts_removable,
                         "DESTROYING Incorrect MAIN_PART parts_removable")

    def test_get_transformed_parts_create(self):
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")

        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.CREATED)

        creating_transformation_model = \
            pro.TransformationModel(identification=None, root_nodes=[main_part_part_transformation_node])

        transformed_parts, destroyed_parts = creating_transformation_model.get_transformed_entities(input_parts=[],
                                                                                                    input_resources=[])

        transformed_part = transformed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type,
                                     part_of=None,
                                     parts=[],
                                     parts_removable=[])

        self.assertEqual(transformed_part.entity_type, right_output_part.entity_type,
                         "DESTROYING Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part.part_of, right_output_part.part_of,
                         "DESTROYING Incorrect MAIN_PART part_of")
        self.assertEqual(transformed_part.parts, right_output_part.parts,
                         "DESTROYING Incorrect MAIN_PART parts")
        self.assertEqual(transformed_part.parts_removable, right_output_part.parts_removable,
                         "DESTROYING Incorrect MAIN_PART parts_removable")

    def test_get_transformed_parts_assembly_super_entity_type(self):
        main_part_super_entity_type = ent.EntityType(identification=None,
                                                     name="super main part")
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part",
                                               super_entity_type=main_part_super_entity_type)
        sub_part_entity_type = ent.EntityType(identification=None,
                                              name="sub part")
        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_super_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        sub_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=sub_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.SUB_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)

        main_part_part_transformation_node.add_child(output_part_part_transformation_node)
        sub_part_part_transformation_node.add_child(output_part_part_transformation_node)
        output_part_part_transformation_node.add_parent(main_part_part_transformation_node,
                                                        sub_part_part_transformation_node)

        assembly_transformation_model = \
            pro.TransformationModel(identification=None,
                                    root_nodes=[main_part_part_transformation_node, sub_part_part_transformation_node])

        main_part_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=main_part_entity_type)
        sub_part_part = ent.Part(identification=None,
                                 name="Sub Part",
                                 entity_type=sub_part_entity_type)

        transformed_parts, destroyed_parts = \
            assembly_transformation_model.get_transformed_entities(input_parts=[main_part_part, sub_part_part],
                                                                   input_resources=[])
        transformed_part = transformed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type,
                                     parts_removable=[True])
        right_sub_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=sub_part_entity_type,
                                  part_of=right_output_part)
        right_output_part.parts = [right_sub_part]

        self.assertEqual(transformed_part.entity_type, right_output_part.entity_type,
                         "ASSEMBLY Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part.parts_removable, right_output_part.parts_removable,
                         "ASSEMBLY Incorrect MAIN_PART parts_removable")
        self.assertEqual(transformed_part.parts[0].entity_type, right_output_part.parts[0].entity_type,
                         "ASSEMBLY Incorrect MAIN_PART SUB_ENTITY EntityType")
        self.assertEqual(transformed_part.parts[0].part_of.entity_type, right_output_part.parts[0].part_of.entity_type,
                         "ASSEMBLY Incorrect MAIN_PART SUB_ENTITY MAIN_PART EntityType")

    def test_get_transformed_parts_support(self):
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")
        support_resource_entity_type = ent.EntityType(identification=None,
                                                      name="support resource")
        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        support_resource_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=support_resource_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.SUPPORT,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)

        main_part_part_transformation_node.add_child(output_part_transformation_node)
        support_resource_part_transformation_node.add_child(output_part_transformation_node)
        output_part_transformation_node.add_parent(main_part_part_transformation_node,
                                                   support_resource_part_transformation_node)

        marriage_transformation_model = \
            pro.TransformationModel(identification=None,
                                    root_nodes=[main_part_part_transformation_node,
                                                support_resource_part_transformation_node])

        main_part_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=main_part_entity_type)
        plant = ent.Plant(identification=None, name="Test Plant", corners=[0], current_time=0, work_calendar=None)
        support_resource_part = ent.ActiveMovingResource(identification=None,
                                                         name="Support Resource",
                                                         entity_type=support_resource_entity_type,
                                                         plant=plant,
                                                         costs_per_second=0,
                                                         position=(0, 0), length=0, width=0,
                                                         orientation=0,
                                                         speed=0, energy_consumption=0,
                                                         energy_capacity=0, energy_level=0,
                                                         allowed_entity_types=[main_part_entity_type],
                                                         capacity_per_entity_type=[1],
                                                         entities_on_transport=[])

        transformed_parts, destroyed_parts = \
            marriage_transformation_model.get_transformed_entities(input_parts=[main_part_part],
                                                                   input_resources=[support_resource_part])
        transformed_part = transformed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type)
        right_support_resource = ent.ActiveMovingResource(identification=None,
                                                          name="Support Resource",
                                                          entity_type=support_resource_entity_type,
                                                          plant=plant,
                                                          costs_per_second=0,
                                                          position=(0, 0), length=0, width=0,
                                                          orientation=0,
                                                          speed=0, energy_consumption=0,
                                                          energy_capacity=0, energy_level=0,
                                                          allowed_entity_types=[main_part_entity_type],
                                                          capacity_per_entity_type=[1],
                                                          entities_on_transport=[right_output_part])
        right_output_part.is_situated_in = right_support_resource

        self.assertEqual(transformed_part.entity_type, right_output_part.entity_type,
                         "SUPPORT Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part.situated_in.entity_type, right_support_resource.entity_type,
                         "SUPPORT Incorrect MAIN_PART SUPPORT EntityType")
        self.assertEqual(transformed_part.situated_in.entities_on_transport[0].name, right_output_part.name,
                         "SUPPORT Incorrect MAIN_PART SUPPORT MAIN_PART EntityType")

    def test_get_transformed_parts_unsupport(self):
        main_part_entity_type = ent.EntityType(identification=None,
                                               name="main part")
        support_resource_entity_type = ent.EntityType(identification=None,
                                                      name="support resource")
        main_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_part_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=main_part_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.MAIN_ENTITY,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)
        output_resource_part_transformation_node = \
            pro.EntityTransformationNode(identification=None,
                                         entity_type=support_resource_entity_type,
                                         amount=1,
                                         quality=1,
                                         transformation_type=pro.EntityTransformationNode.TransformationTypes.UNSUPPORT,
                                         io_behaviour=pro.EntityTransformationNode.IoBehaviours.EXIST)

        main_part_part_transformation_node.add_child(output_part_part_transformation_node,
                                                     output_resource_part_transformation_node)
        output_part_part_transformation_node.add_parent(main_part_part_transformation_node)
        output_resource_part_transformation_node.add_parent(main_part_part_transformation_node)

        divorce_transformation_model = \
            pro.TransformationModel(identification=None,
                                    root_nodes=[main_part_part_transformation_node])

        main_part_part = ent.Part(identification=None,
                                  name="Main Part",
                                  entity_type=main_part_entity_type)
        plant = ent.Plant(identification=None, name="Test Plant", corners=[0], current_time=0, work_calendar=None)
        support_resource = ent.ActiveMovingResource(identification=None,
                                                    name="Support Resource",
                                                    entity_type=support_resource_entity_type,
                                                    plant=plant,
                                                    costs_per_second=0,
                                                    position=(0, 0), length=0, width=0,
                                                    orientation=0,
                                                    speed=0, energy_consumption=0,
                                                    energy_capacity=0, energy_level=0,
                                                    allowed_entity_types=[main_part_entity_type],
                                                    capacity_per_entity_type=[1],
                                                    entities_on_transport=[main_part_part])
        main_part_part.is_situated_in = support_resource

        transformed_parts, destroyed_parts = divorce_transformation_model.get_transformed_entities(
            input_parts=[main_part_part], input_resources=[])

        transformed_part = transformed_parts[0][0]

        right_output_part = ent.Part(identification=None,
                                     name="Main Part",
                                     entity_type=main_part_entity_type)
        right_support_resource = ent.ActiveMovingResource(identification=None,
                                                          name="Support Resource",
                                                          entity_type=support_resource_entity_type,
                                                          plant=plant,
                                                          costs_per_second=0,
                                                          position=(0, 0), length=0, width=0,
                                                          orientation=0,
                                                          speed=0, energy_consumption=0,
                                                          energy_capacity=0, energy_level=0,
                                                          allowed_entity_types=[main_part_entity_type],
                                                          capacity_per_entity_type=[1], entities_on_transport=[])

        self.assertEqual(transformed_part.entity_type, right_output_part.entity_type,
                         "UNSUPPORT Incorrect MAIN_PART EntityType")
        self.assertEqual(transformed_part.situated_in, None,
                         "UNSUPPORT Incorrect MAIN_PART situated_in")


if __name__ == '__main__':
    unittest.main()
