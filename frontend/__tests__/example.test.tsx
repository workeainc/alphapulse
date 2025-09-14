import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'

describe('Example Test', () => {
  it('should pass basic test', () => {
    expect(true).toBe(true)
  })

  it('should render a simple component', () => {
    const TestComponent = () => <div data-testid="test-component">Hello World</div>
    
    render(<TestComponent />)
    
    const element = screen.getByTestId('test-component')
    expect(element).toBeInTheDocument()
    expect(element).toHaveTextContent('Hello World')
  })
})
