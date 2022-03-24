from typing import Any


class Node:
    value: Any
    next: 'Node'

    def __init__(self, value: Any):
        self.value = value
        self.next = None

class LinkedList:
    head: Node
    tail: Node

    def __init__(self):
        self.head = None
        self.tail = None

    def push(self, value: Any) -> None:
        node = Node(value)
        node.next = self.head
        self.head = node

    def append(self, value: Any) -> None:
        node = Node(value)
        if self.head is None:
            self.head = node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = node

    def node(self, at: int) -> Node:
            current = self.head
            count = 0
            while current is not None:
                if count == at:
                    return current.value
                count += 1
                current = current.next
            if current is None:
                print("Brak takiego indexu")

    def insert(self, value: Any, after: Node) -> None:
        if after != self.tail:
            temp: Node = after.next
            after.next = Node(value)
            after.next.next = temp
        else:
            self.append(value)

    def pop(self) -> Any:
        node = self.head
        self.head = node.next
        return node

    def remove_last(self) -> Any:
        if self.head is not None:
            if self.head.next is None:
                self.head = None
            else:
                current = self.head
                while current.next.next is not None:
                    current = current.next
                node = current.next
                current.next = None
                node = None

    def remove(self, after: Node) -> Any:
        if after == 0 and self.head is not None:
            nodetodel = self.head
            self.head = self.head.next
            nodetodel = None
        else:
            temp = self.head
            for i in range(0, after - 1):
                if temp is not None:
                    temp = temp.next
            if temp is not None and temp.next is not None:
                node = temp.next
                temp.next = temp.next.next
                node = None
            else:
                print("\nNie ma takiego")

    def __str__(self):
        new_node = self.head
        text = ""
        while new_node.next:
            text += str(new_node.value) + " -> "
            new_node = new_node.next
        text += str(new_node.value)
        return text

    def __len__(self):
        length: int = 0
        temp: Node = self.head
        if self.head is None:
            return 0
        else:
            while temp != self.tail:
                temp = temp.next
                length = length + 1

        return length

class Stack:
    _storage: LinkedList

    def __init__(self):
        self._storage = LinkedList()

    def push(self, element: Any) -> None:
        self._storage.append(element)

    def pop(self) -> Any:
        if len(self._storage) != 0:
            return self._storage.remove_last()

    def __len__(self):
        return len(self._storage)

    def __str__(self):
        stack = self._storage.head
        text = ""
        for i in range(0, len(self._storage)):
            text += "| "  + str(stack.value) + " |" + "\n"
            stack = stack.next
        return text

class Queue:
    _storage: LinkedList()

    def __init__(self):
        self._storage = LinkedList()

    def peek(self) -> Any:
        return self._storage.head.value

    def enqueue(self, element: Any) -> None:
        self._storage.append(element)

    def dequeue(self) -> Any:
        return self._storage.pop()

    def __str__(self):
        que = self._storage.head
        text = ""
        while que.next:
            text += str(que.value) + " -> "
            que = que.next
        text += str(que.value)
        return text

    def __len__(self):
        return len(self._storage)


my_list = LinkedList()
my_list.push(6)
print((my_list.__str__()))
my_list.push(5)
print((my_list.__str__()))
my_list.push(4)
print((my_list.__str__()))
my_list.push(3)
my_list.push(2)
print((my_list.__str__()))
my_list.append(7)
print((my_list.__str__()))
print(my_list.node(2))
my_list.pop()
print((my_list.__str__()))
my_list.remove_last()
print((my_list.__str__()))
my_list.remove(2)
print((my_list.__str__()))
my_list.insert(9, my_list.head.next)
print((my_list.__str__()))
print((my_list.__len__()))



my_stack = Stack()
my_stack.push(1)
print((my_stack.__str__()))
my_stack.push(3)
print((my_stack.__str__()))
my_stack.push(22)
print((my_stack.__str__()))
my_stack.pop()
print((my_stack.__str__()))
print((my_stack.__len__()))



my_queue = Queue()
my_queue.enqueue("pani w ciazy")
print((my_queue.__str__()))
my_queue.enqueue("petent1")
my_queue.enqueue("petent2")
print((my_queue.__str__()))
my_queue.enqueue("petent3")
print((my_queue.__str__()))
my_queue.dequeue()
print((my_queue.__str__()))
print((my_queue.__len__()))
print(my_queue.peek())



